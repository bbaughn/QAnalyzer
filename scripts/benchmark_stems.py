#!/usr/bin/env python3
"""Benchmark stem-based harmonic onset detection for bars_percussion.

Separates each track's first CLIP_SEC seconds with Demucs htdemucs, measures
per-beat RMS of the 'other' (melodic/harmonic) stem, then applies a simple
threshold to estimate bars_percussion.

Hypothesis: the 'other' stem is near-silent during a drums-only intro and
jumps sharply when harmonic content enters — making detection trivial compared
to the current HPSS + attack/sustain approach.

Stem RMS profiles are cached in test_tracks/stem_cache/ so re-runs are fast.
Separation only — no model training, no gradient computation.

Usage:
    PYTHONPATH=. .venv/bin/python3.12 scripts/benchmark_stems.py
    PYTHONPATH=. .venv/bin/python3.12 scripts/benchmark_stems.py --rerun
    PYTHONPATH=. .venv/bin/python3.12 scripts/benchmark_stems.py --tracks blooms tikken
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
EXPECTATIONS = ROOT / "test_tracks" / "expectations.yaml"
CACHE_DIR = ROOT / "test_tracks" / "features_cache"
STEM_CACHE_DIR = ROOT / "test_tracks" / "stem_cache"
STEM_CACHE_DIR.mkdir(exist_ok=True)

CLIP_SEC = 90.0       # only separate the first N seconds (covers ~32 bars at 90 BPM)
ONSET_FRAC = 0.05     # 'other' stem onset threshold: fraction of max RMS

sys.path.insert(0, str(ROOT))
import app.services.analysis as analysis


# ---------------------------------------------------------------------------
# Stem separation
# ---------------------------------------------------------------------------

def separate_other_stem(wav_path: Path, clip_sec: float = CLIP_SEC) -> tuple[np.ndarray, int]:
    """Return (other_stem_mono, sample_rate) for the first clip_sec seconds."""
    import librosa
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    y, sr = librosa.load(str(wav_path), sr=44100, mono=False, duration=clip_sec)
    if y.ndim == 1:
        y = np.stack([y, y])  # mono → stereo [2, T]

    model = get_model("htdemucs")
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    audio_tensor = torch.from_numpy(y).unsqueeze(0).float().to(device)  # [1, 2, T]

    with torch.no_grad():
        sources = apply_model(model, audio_tensor, progress=False)
    # sources: [batch=1, n_sources, channels=2, T]

    other_idx = model.sources.index("other")
    other_stereo = sources[0, other_idx].cpu().numpy()  # [2, T]
    other_mono = other_stereo.mean(axis=0)              # [T]

    return other_mono, sr


# ---------------------------------------------------------------------------
# Per-beat RMS and onset detection
# ---------------------------------------------------------------------------

def compute_beat_rms(audio: np.ndarray, sr: int, beat_times: np.ndarray) -> np.ndarray:
    """RMS energy of audio in each beat window. Shape: (n_beats-1,)."""
    rms = []
    n = len(beat_times)
    for i in range(n - 1):
        t0 = int(beat_times[i] * sr)
        t1 = min(int(beat_times[i + 1] * sr), len(audio))
        if t0 >= len(audio) or t0 >= t1:
            break
        seg = audio[t0:t1]
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
    return np.array(rms)


def detect_onset_beat(beat_rms: np.ndarray, threshold_frac: float = ONSET_FRAC) -> int:
    """Return beat index where 'other' stem first exceeds threshold_frac × max RMS."""
    if beat_rms.size == 0:
        return 0
    threshold = beat_rms.max() * threshold_frac
    for i, v in enumerate(beat_rms):
        if v >= threshold:
            return i
    return 0


def detect_onset_step_change(
    beat_rms: np.ndarray,
    window: int = 4,
    ratio: float = 8.0,
    min_fwd: float = 0.03,
) -> int:
    """Return onset as first beat where forward energy sustainedly exceeds backward energy.

    Slides a window of `window` beats on each side. Returns the first beat i where:
      - mean(beat_rms[i : i+window]) > min_fwd * max_rms  (forward is non-trivial)
      - mean(beat_rms[i : i+window]) > mean(beat_rms[i-window : i]) * ratio

    For bars_percussion=0 tracks: beat 0 has no backward window (mean→0), so
    forward/backward → ∞ and fires immediately if forward > min_fwd.
    For long intros: the forward window stays low until the harmonic burst, then
    the ratio spikes sharply at the onset beat.
    """
    if beat_rms.size == 0:
        return 0

    max_rms = beat_rms.max()
    if max_rms < 1e-9:
        return 0

    norm = beat_rms / max_rms

    for i in range(len(norm)):
        fwd = norm[i : i + window]
        bwd = norm[max(0, i - window) : i]

        fwd_mean = float(fwd.mean()) if fwd.size > 0 else 0.0
        bwd_mean = float(bwd.mean()) if bwd.size > 0 else 0.0

        if fwd_mean >= min_fwd and fwd_mean > bwd_mean * ratio:
            return i

    return 0


def detect_onset_median_relative(
    beat_rms: np.ndarray,
    window: int = 4,
    intro_frac: float = 0.4,
    onset_frac: float = 0.5,
) -> int:
    """Median-relative onset detector.

    Step 1 — Is there an intro at all?
      Compute median energy of the whole clip. If the first `window` beats
      already average ≥ intro_frac × median, harmonic content is present
      from beat 0 → return 0.

    Step 2 — Find onset.
      Scan forward; return the first beat where the `window`-beat forward mean
      ≥ onset_frac × clip median. This is immune to:
        - Beat-0 ratio=∞ (compared against median, not empty backward window)
        - Late-clip peaks dominating (threshold is absolute, not relative to max)
    """
    if beat_rms.size < window:
        return 0

    clip_median = float(np.median(beat_rms))
    if clip_median < 1e-9:
        return 0

    # Step 1: check if there's actually a percussion intro
    first_window_mean = float(beat_rms[:window].mean())
    if first_window_mean >= intro_frac * clip_median:
        return 0  # harmonic content from beat 0

    # Step 2: find the onset
    onset_threshold = onset_frac * clip_median
    for i in range(len(beat_rms)):
        fwd = beat_rms[i : i + window]
        if fwd.size > 0 and float(fwd.mean()) >= onset_threshold:
            return i

    return 0


def detect_onset_max_step(beat_rms: np.ndarray, window: int = 4) -> int:
    """Return the beat with the largest forward/backward energy ratio.

    Unlike step_change (which returns the *first* beat exceeding a threshold),
    this finds the *sharpest* single step across the whole clip — useful when
    the threshold-based approach fires too early on low-level bleed.
    """
    if beat_rms.size < window * 2:
        return 0

    max_rms = beat_rms.max()
    if max_rms < 1e-9:
        return 0

    norm = beat_rms / max_rms
    best_i, best_ratio = 0, 0.0

    for i in range(1, len(norm)):
        fwd = norm[i : i + window]
        bwd = norm[max(0, i - window) : i]
        fwd_mean = float(fwd.mean()) if fwd.size > 0 else 0.0
        bwd_mean = float(bwd.mean()) if bwd.size > 0 else 0.0
        r = fwd_mean / (bwd_mean + 1e-6)
        if r > best_ratio:
            best_ratio, best_i = r, i

    # If the sharpest step is at beat 0 area with no real intro, clamp to 0
    # by checking whether beat 0's forward mean is already close to the peak.
    fwd0 = norm[0:window].mean() if len(norm) >= window else norm.mean()
    if fwd0 >= norm.mean() * 0.8:
        return 0

    return best_i


def _round_bars(bars: float) -> int:
    """Snap to nearest standard bar count (mirrors analysis.py rounding)."""
    if bars < 0.9:
        return 0
    breakpoints = [1, 2, 3, 4, 8, 16, 24, 32, 40, 48, 64]
    return min(breakpoints, key=lambda b: abs(b - bars))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_stem_cache(stem: str) -> dict | None:
    path = STEM_CACHE_DIR / f"{stem}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    d["beat_rms"] = np.array(d["beat_rms"])
    return d


def save_stem_cache(stem: str, data: dict) -> None:
    out = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    (STEM_CACHE_DIR / f"{stem}.json").write_text(json.dumps(out))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(rerun: bool = False, track_filter: list[str] | None = None) -> None:
    tracks = yaml.safe_load(EXPECTATIONS.read_text())["tracks"]

    # Only process tracks that have a bars_percussion expectation and a cached feature file
    candidates = []
    for t in tracks:
        exp = t.get("expected", {})
        if "bars_percussion" not in exp:
            continue
        wav = ROOT / t["file"]
        cache = CACHE_DIR / (wav.stem + ".json")
        if not wav.exists() or not cache.exists():
            continue
        if track_filter and wav.stem not in track_filter and t["title"] not in track_filter:
            continue
        candidates.append(t)

    stem_correct = algo_correct = thresh_correct = max_correct = median_correct = 0
    total = 0

    for track in candidates:
        title = track["title"]
        wav = ROOT / track["file"]
        expected_bars = track["expected"]["bars_percussion"]
        cache = CACHE_DIR / (wav.stem + ".json")

        # Beat times from feature cache, clipped to CLIP_SEC
        features_data = json.loads(cache.read_text())
        beat_times = np.array(features_data.get("beat_times", []))
        beat_times = beat_times[beat_times <= CLIP_SEC]
        if len(beat_times) < 2:
            print(f"\n  {title}: not enough beats in {CLIP_SEC}s clip — skipping")
            continue

        # Current algorithm's result
        features = analysis.load_features(str(cache))
        result = analysis.interpret_features(features)
        algo_bars = result.get("global", {}).get("bars_percussion_rounded", "?")

        # Stem RMS (cached or freshly separated)
        cached = None if rerun else load_stem_cache(wav.stem)
        if cached is None:
            print(f"\n  [{wav.stem}] separating stems (first {CLIP_SEC:.0f}s)…", flush=True)
            other_mono, sr = separate_other_stem(wav)
            beat_rms = compute_beat_rms(other_mono, sr, beat_times)
            save_stem_cache(wav.stem, {"beat_rms": beat_rms})
            print(f"  [{wav.stem}] done — cached to stem_cache/{wav.stem}.json")
        else:
            beat_rms = cached["beat_rms"]

        # Detect onset — four methods
        beat_thresh  = detect_onset_beat(beat_rms)
        beat_step    = detect_onset_step_change(beat_rms)
        beat_max     = detect_onset_max_step(beat_rms)
        beat_median  = detect_onset_median_relative(beat_rms)

        bars_thresh  = _round_bars(beat_thresh / 4.0)
        bars_step    = _round_bars(beat_step / 4.0)
        bars_max     = _round_bars(beat_max / 4.0)
        bars_median  = _round_bars(beat_median / 4.0)

        total += 1
        algo_ok    = algo_bars == expected_bars
        thresh_ok  = bars_thresh == expected_bars
        step_ok    = bars_step == expected_bars
        max_ok     = bars_max == expected_bars
        median_ok  = bars_median == expected_bars
        if algo_ok:    algo_correct += 1
        if thresh_ok:  thresh_correct += 1
        if step_ok:    stem_correct += 1
        if max_ok:     max_correct += 1
        if median_ok:  median_correct += 1

        print(f"\n{'─' * 68}")
        print(f"  {title}")
        print(f"  expected={expected_bars}  algo={'✓' if algo_ok else '✗'}{algo_bars}")
        print(f"  thresh={'✓' if thresh_ok else '✗'}{bars_thresh} (beat {beat_thresh})"
              f"  step={'✓' if step_ok else '✗'}{bars_step} (beat {beat_step})"
              f"  max_step={'✓' if max_ok else '✗'}{bars_max} (beat {beat_max})"
              f"  median={'✓' if median_ok else '✗'}{bars_median} (beat {beat_median})")

        # ASCII energy profile — first 48 beats (12 bars)
        max_rms = beat_rms.max() if beat_rms.max() > 1e-9 else 1.0
        norm = beat_rms / max_rms

        print()
        print(f"  Beat  Bar   Norm   0%{'':>19}50%{'':>16}100%")
        print(f"  {'─' * 62}")
        display_beats = min(len(norm), 48)
        for i in range(display_beats):
            bar_num = i // 4 + 1
            beat_in_bar = i % 4 + 1
            bar_label = f"{bar_num}" if beat_in_bar == 1 else ""
            beat_label = f"{beat_in_bar}"
            filled = int(norm[i] * 40)
            bar_str = "█" * filled + "░" * (40 - filled)
            markers = []
            if i == beat_thresh:  markers.append("T")
            if i == beat_step:    markers.append("S")
            if i == beat_max:     markers.append("M")
            if i == beat_median:  markers.append("R")
            tag = f" ◄ {'/'.join(markers)}" if markers else ""
            print(f"  {i:>3}   {bar_label:>3}.{beat_label}  {norm[i]:.3f}  {bar_str}{tag}")
        if len(norm) > 48:
            print(f"  … ({len(norm) - 48} more beats not shown)")

    print(f"\n{'═' * 68}")
    print(f"  bars_percussion accuracy over {total} tracks:")
    print(f"    algo        : {algo_correct}/{total}")
    print(f"    stem-thresh : {thresh_correct}/{total}  (first beat ≥ 5% of max)")
    print(f"    stem-step   : {stem_correct}/{total}  (fwd/bwd ratio ≥ 8×, window=4)")
    print(f"    stem-maxstep: {max_correct}/{total}  (sharpest single step)")
    print(f"    stem-median : {median_correct}/{total}  (median-relative, intro_frac=0.4, onset_frac=0.5)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-separation even if cached")
    parser.add_argument("--tracks", nargs="+",
                        help="Limit to specific track stems or titles (e.g. blooms tikken)")
    args = parser.parse_args()
    run(rerun=args.rerun, track_filter=args.tracks)
