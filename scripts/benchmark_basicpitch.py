#!/usr/bin/env python3
"""Benchmark Basic-Pitch MIDI-derived key detection vs chroma-based detection.

Tests four progressively richer approaches on the MIDI data:

  midi_raw      — profile-match on global MIDI PC histogram, no heuristics
  midi_root     — add argmax/Krumhansl 95% root heuristic + P5 correction,
                  then constrained profile-match
  midi_win      — full per-beat-window analysis (same structure as the chroma
                  pipeline) using MIDI PC histograms per window
  midi_full     — midi_win + segment-level dorian/mixolydian refinement
                  (m6 vs b6 from MIDI PC, not chroma)

All compared against the current chroma pipeline result and expected key.

MIDI note events are cached in test_tracks/midi_cache/ so re-runs are fast.

Usage:
    PYTHONPATH=. python3.12 scripts/benchmark_basicpitch.py
    PYTHONPATH=. python3.12 scripts/benchmark_basicpitch.py --rerun
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = Path(__file__).parent.parent
EXPECTATIONS = ROOT / "test_tracks" / "expectations.yaml"
CACHE_DIR = ROOT / "test_tracks" / "features_cache"
MIDI_CACHE_DIR = ROOT / "test_tracks" / "midi_cache"
MIDI_CACHE_DIR.mkdir(exist_ok=True)

ONNX_MODEL = Path(os.path.expanduser(
    "~/.local/venvs/qanalyzer/lib/python3.12/site-packages/"
    "basic_pitch/saved_models/icassp_2022/nmp.onnx"
))

sys.path.insert(0, str(ROOT))
import app.services.analysis as analysis
from app.config import settings

KEYS = analysis.KEYS
ENHARMONIC = {
    "C#": "Db", "Db": "C#", "D#": "Eb", "Eb": "D#",
    "F#": "Gb", "Gb": "F#", "G#": "Ab", "Ab": "G#",
    "A#": "Bb", "Bb": "A#",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def roots_match(a, b):
    if a is None or b is None:
        return False
    return a == b or ENHARMONIC.get(a) == b


def transcribe(wav_path: Path) -> list[tuple]:
    from basic_pitch.inference import predict
    _, _, note_events = predict(str(wav_path), str(ONNX_MODEL))
    return [(float(on), float(off), int(p), float(v))
            for on, off, p, v, _ in note_events]


def load_midi_cache(stem: str) -> list[tuple] | None:
    path = MIDI_CACHE_DIR / f"{stem}.json"
    return [tuple(e) for e in json.loads(path.read_text())] if path.exists() else None


def save_midi_cache(stem: str, events: list[tuple]) -> None:
    (MIDI_CACHE_DIR / f"{stem}.json").write_text(json.dumps(events))


# ---------------------------------------------------------------------------
# MIDI PC histogram utilities
# ---------------------------------------------------------------------------

def global_pc_hist(events: list[tuple]) -> np.ndarray:
    """12-bin PC histogram weighted by duration × velocity over all events."""
    h = np.zeros(12)
    for onset, offset, pitch, velocity in events:
        h[pitch % 12] += (offset - onset) * velocity
    return h


def per_beat_pc_hists(events: list[tuple], beat_times: np.ndarray) -> np.ndarray:
    """Shape (12, n_beats-1): MIDI PC energy per beat window.

    Each column is the sum of (overlap_duration × velocity) for notes whose
    [onset, offset) interval overlaps the beat window [t_i, t_{i+1}).
    """
    n = beat_times.size
    if n < 2:
        return np.zeros((12, 1))
    cols = n - 1
    hists = np.zeros((12, cols))
    for onset, offset, pitch, velocity in events:
        pc = pitch % 12
        # find beat windows this note overlaps
        lo = np.searchsorted(beat_times, onset, side="right") - 1
        hi = np.searchsorted(beat_times, offset, side="left")
        lo = max(lo, 0)
        hi = min(hi, cols)
        for c in range(lo, hi):
            win_start = beat_times[c]
            win_end = beat_times[min(c + 1, n - 1)]
            overlap = min(offset, win_end) - max(onset, win_start)
            if overlap > 0:
                hists[pc, c] += overlap * velocity
    return hists


# ---------------------------------------------------------------------------
# Root detection from MIDI histogram (mirrors _segment_key_timeline logic)
# ---------------------------------------------------------------------------

def midi_global_root(midi_hist: np.ndarray) -> tuple[int, bool]:
    """Return (global_root_idx, p5_corrected) using argmax/Krumhansl + P5."""
    argmax_idx = int(np.argmax(midi_hist))
    est = analysis._key_from_chroma(midi_hist)
    krumhansl_idx = KEYS.index(est.key)

    if midi_hist[krumhansl_idx] >= midi_hist[argmax_idx] * 0.95:
        global_root_idx = krumhansl_idx
    else:
        global_root_idx = argmax_idx

    p5_corrected = False
    if krumhansl_idx == global_root_idx and est.mode in {"minor", "locrian"}:
        b2_idx  = (global_root_idx + 1) % 12
        nat2_idx = (global_root_idx + 2) % 12
        if midi_hist[b2_idx] > midi_hist[nat2_idx] * 1.25:
            cand = (global_root_idx + 5) % 12
            h_n = midi_hist / (np.linalg.norm(midi_hist) + 1e-9)
            minor_norm = next(p for n, p in analysis._MODE_PROFILES_NORM if n == "minor")
            if np.dot(h_n, np.roll(minor_norm, cand)) > np.dot(h_n, np.roll(minor_norm, global_root_idx)) * 0.97:
                global_root_idx = cand
                p5_corrected = True

    return global_root_idx, p5_corrected


# ---------------------------------------------------------------------------
# Mode 1: raw profile match on global MIDI histogram
# ---------------------------------------------------------------------------

def midi_raw_key(midi_hist: np.ndarray) -> tuple[str, str]:
    est = analysis._key_from_chroma(midi_hist)
    return est.key, est.mode


# ---------------------------------------------------------------------------
# Mode 2: root heuristics + constrained mode
# ---------------------------------------------------------------------------

def midi_root_key(midi_hist: np.ndarray) -> tuple[str, str]:
    root_idx, p5 = midi_global_root(midi_hist)
    est = analysis._key_from_chroma(midi_hist, constrain_root=root_idx)
    mode = "minor" if p5 else est.mode
    return KEYS[root_idx], mode


# ---------------------------------------------------------------------------
# Mode 3+4: per-window analysis (full pipeline equivalent)
# ---------------------------------------------------------------------------

def _dorian_refine_midi(segments: list[dict], beat_hists: np.ndarray,
                         beat_times: np.ndarray, threshold: float = 1.02) -> list[dict]:
    """Mirror of _apply_segment_dorian_refinement but uses MIDI beat hists."""
    for seg in segments:
        if seg.get("mode") != "minor":
            continue
        key_idx = KEYS.index(seg["key"])
        mask = (beat_times[:-1] >= seg["start"]) & (beat_times[:-1] <= seg["end"])
        cols = np.where(mask)[0]
        if cols.size == 0:
            continue
        seg_hist = np.mean(beat_hists[:, cols], axis=1)
        m6_idx = (key_idx + 9) % 12
        b6_idx = (key_idx + 8) % 12
        if seg_hist[m6_idx] > seg_hist[b6_idx] * threshold:
            seg["mode"] = "dorian"
    return segments


def midi_windowed_key(events: list[tuple], beat_times: np.ndarray,
                       dorian_refine: bool = True) -> tuple[str | None, str | None]:
    """Full per-window MIDI pipeline returning (root, mode) of first segment."""
    beat_hists = per_beat_pc_hists(events, beat_times)
    midi_hist = global_pc_hist(events)

    root_idx, p5 = midi_global_root(midi_hist)
    window_margin = 1.0 if p5 else 0.015

    # Build raw per-window key segments
    raw: list[dict] = []
    for c in range(beat_hists.shape[1]):
        h = beat_hists[:, c]
        if h.sum() < 1e-9:
            continue
        est_free = analysis._key_from_chroma(h)
        if root_idx is not None:
            est_constrained = analysis._key_from_chroma(h, constrain_root=root_idx)
            if est_free.confidence - est_constrained.confidence <= window_margin:
                est = est_constrained
            else:
                est = est_free
        else:
            est = est_free
        t = float(beat_times[c])
        t_end = float(beat_times[min(c + 1, beat_times.size - 1)])
        raw.append({"start": t, "end": t_end, "key": est.key,
                    "mode": est.mode, "confidence": float(est.confidence)})

    if not raw:
        return None, None

    if p5:
        cand_key = KEYS[root_idx]
        for seg in raw:
            if seg["key"] == cand_key and seg["mode"] != "minor":
                seg["mode"] = "minor"

    # Coalesce same-label neighbours
    coalesced: list[dict] = []
    for seg in raw:
        if coalesced and coalesced[-1]["key"] == seg["key"] and coalesced[-1]["mode"] == seg["mode"]:
            coalesced[-1]["end"] = seg["end"]
            coalesced[-1]["confidence"] = (coalesced[-1]["confidence"] + seg["confidence"]) / 2
        else:
            coalesced.append(dict(seg))

    # Confirm segments (persist + confidence margin)
    persist = settings.key_change_min_persist_windows
    conf_margin = settings.key_change_conf_margin
    min_conf = settings.key_change_min_confidence

    confirmed: list[dict] = [dict(coalesced[0])]
    i = 1
    while i < len(coalesced):
        seg = coalesced[i]
        current = confirmed[-1]
        if current["key"] == seg["key"] and current["mode"] == seg["mode"]:
            current["end"] = seg["end"]
            current["confidence"] = (current["confidence"] + seg["confidence"]) / 2
            i += 1
            continue
        run, j = [], i
        while j < len(coalesced) and coalesced[j]["key"] == seg["key"] and coalesced[j]["mode"] == seg["mode"]:
            run.append(coalesced[j]); j += 1
        if run:
            run_conf = float(np.mean([r["confidence"] for r in run]))
            if len(run) >= persist and run_conf >= max(min_conf, current["confidence"] + conf_margin):
                confirmed.append({"start": run[0]["start"], "end": run[-1]["end"],
                                   "key": run[0]["key"], "mode": run[0]["mode"],
                                   "confidence": run_conf})
            else:
                current["end"] = run[-1]["end"]
                current["confidence"] = (current["confidence"] + run_conf) / 2
        i = j

    if dorian_refine:
        confirmed = _dorian_refine_midi(confirmed, beat_hists, beat_times)

    first = confirmed[0]
    return first["key"], first["mode"]


# ---------------------------------------------------------------------------
# Pipeline key (full chroma pipeline)
# ---------------------------------------------------------------------------

def pipeline_key(cache_path: Path) -> tuple[str | None, str | None]:
    features = analysis.load_features(str(cache_path))
    result = analysis.interpret_features(features)
    secs = result.get("sections", [])
    if not secs or secs[0].get("key") is None:
        return None, None
    return secs[0]["key"], secs[0]["mode"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(rerun: bool = False) -> None:
    tracks = yaml.safe_load(EXPECTATIONS.read_text())["tracks"]

    rows = []
    for track in tracks:
        title = track["title"]
        wav = ROOT / track["file"]
        cache = CACHE_DIR / (wav.stem + ".json")
        expected = track.get("expected", {})

        if not wav.exists() or not cache.exists():
            continue
        if expected.get("no_key") or expected.get("no_tempo"):
            continue
        exp_key = expected.get("key")
        if not exp_key:
            continue

        exp_root, exp_mode = exp_key["root"], exp_key["mode"]

        # MIDI events
        midi_events = None if rerun else load_midi_cache(wav.stem)
        if midi_events is None:
            print(f"  Transcribing {title}…", flush=True)
            midi_events = transcribe(wav)
            save_midi_cache(wav.stem, midi_events)

        # Beat times from features cache
        features_data = json.loads(cache.read_text())
        beat_times = np.array(features_data.get("beat_times", []))

        midi_hist = global_pc_hist(midi_events)

        r1_root, r1_mode = midi_raw_key(midi_hist)
        r2_root, r2_mode = midi_root_key(midi_hist)
        r3_root, r3_mode = midi_windowed_key(midi_events, beat_times, dorian_refine=False)
        r4_root, r4_mode = midi_windowed_key(midi_events, beat_times, dorian_refine=True)
        pip_root, pip_mode = pipeline_key(cache)

        def ok(root, mode):
            return roots_match(root, exp_root), (mode == exp_mode and roots_match(root, exp_root))

        rows.append({
            "title": title,
            "exp_root": exp_root, "exp_mode": exp_mode,
            "r1": (r1_root, r1_mode), "r1_ok": ok(r1_root, r1_mode),
            "r2": (r2_root, r2_mode), "r2_ok": ok(r2_root, r2_mode),
            "r3": (r3_root, r3_mode), "r3_ok": ok(r3_root, r3_mode),
            "r4": (r4_root, r4_mode), "r4_ok": ok(r4_root, r4_mode),
            "pip": (pip_root, pip_mode), "pip_ok": ok(pip_root, pip_mode),
        })

    def fmt(root, mode, root_ok, mode_ok):
        return f"{'✓' if root_ok else '✗'}{root or '?'} {'✓' if mode_ok else '✗'}{mode or '?'}"

    cols = ["r1", "r2", "r3", "r4", "pip"]
    labels = ["midi_raw", "midi_root", "midi_win", "midi_full", "pipeline"]
    W = 16

    print(f"\n  {'Track':<40} {'Expected':<14}", end="")
    for lb in labels:
        print(f"  {lb:<{W}}", end="")
    print()
    print("  " + "─" * (40 + 14 + (W + 2) * len(cols) + 2))

    totals_root = {c: 0 for c in cols}
    totals_mode = {c: 0 for c in cols}
    n = len(rows)

    for r in rows:
        exp = f"{r['exp_root']} {r['exp_mode']}"
        print(f"  {r['title']:<40} {exp:<14}", end="")
        for c in cols:
            root_ok, mode_ok = r[f"{c}_ok"]
            root, mode = r[c]
            s = fmt(root, mode, root_ok, mode_ok)
            print(f"  {s:<{W}}", end="")
            if root_ok: totals_root[c] += 1
            if mode_ok: totals_mode[c] += 1
        print()

    print("  " + "─" * (40 + 14 + (W + 2) * len(cols) + 2))
    print(f"  {'root':<40} {n:<14}", end="")
    for c in cols:
        print(f"  {totals_root[c]}/{n:<{W-2}}", end="")
    print()
    print(f"  {'mode':<40} {'':<14}", end="")
    for c in cols:
        print(f"  {totals_mode[c]}/{n:<{W-2}}", end="")
    print()

    print()
    pip_root_n = totals_root["pip"]
    pip_mode_n = totals_mode["pip"]
    for c, lb in zip(cols[:-1], labels[:-1]):
        rg = [r["title"] for r in rows if r[f"{c}_ok"][0] and not r["pip_ok"][0]]
        rl = [r["title"] for r in rows if not r[f"{c}_ok"][0] and r["pip_ok"][0]]
        mg = [r["title"] for r in rows if r[f"{c}_ok"][1] and not r["pip_ok"][1]]
        ml = [r["title"] for r in rows if not r[f"{c}_ok"][1] and r["pip_ok"][1]]
        print(f"  {lb} vs pipeline  root: {len(rg)-len(rl):+d}  mode: {len(mg)-len(ml):+d}")
        if rg: print(f"    root gains : {rg}")
        if rl: print(f"    root losses: {rl}")
        if mg: print(f"    mode gains : {mg}")
        if ml: print(f"    mode losses: {ml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(rerun=args.rerun)
