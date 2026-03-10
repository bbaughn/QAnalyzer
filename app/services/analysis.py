from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import librosa
import numpy as np

from app.config import settings

KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Modes that share a minor third above the root (b3 in scale degree terms).
# Within this family, transitions like minor↔dorian are tonal colour shifts,
# not structural modulations, and should be coalesced.
_MINOR_FAMILY: frozenset[str] = frozenset({"minor", "dorian", "phrygian", "aeolian", "locrian"})
_MAJOR_FAMILY: frozenset[str] = frozenset({"major", "lydian", "mixolydian", "ionian"})

# Krumhansl-style chroma profiles for all 7 diatonic modes.
# Each array starts on C; np.roll(profile, n) gives the profile rooted on KEYS[n].
# Weights derived from Krumhansl (1990) for major/minor and adapted by scale-degree
# importance (tonic > 5th > 3rd > other scale tones > chromatic non-members) for the
# remaining modes.
MODE_PROFILES: list[tuple[str, np.ndarray]] = [
    ("major",      np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])),
    # Dorian: 1 2 b3 4 5 6 b7  (raised 6th vs natural minor)
    ("dorian",     np.array([6.33, 2.30, 3.52, 5.38, 2.30, 3.53, 2.30, 4.75, 2.30, 3.66, 3.34, 2.30])),
    # Phrygian: 1 b2 b3 4 5 b6 b7  (lowered 2nd is the defining colour)
    ("phrygian",   np.array([6.33, 5.00, 2.30, 5.38, 2.30, 3.53, 2.30, 4.75, 3.98, 2.30, 3.34, 2.30])),
    # Lydian: 1 2 3 #4 5 6 7  (raised 4th vs major)
    ("lydian",     np.array([6.35, 2.23, 3.48, 2.23, 4.38, 2.23, 4.00, 5.19, 2.39, 3.66, 2.29, 3.50])),
    # Mixolydian: 1 2 3 4 5 6 b7  (lowered 7th vs major)
    ("mixolydian", np.array([6.35, 2.23, 3.48, 2.23, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 3.34, 2.29])),
    ("minor",      np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])),
    # Locrian: 1 b2 b3 4 b5 b6 b7  (diminished 5th)
    ("locrian",    np.array([6.33, 4.50, 2.30, 5.00, 2.30, 3.53, 4.50, 2.30, 3.98, 2.30, 3.34, 2.30])),
]

# Pre-normalise so _key_from_chroma only needs one norm per profile.
_MODE_PROFILES_NORM: list[tuple[str, np.ndarray]] = [
    (name, profile / (np.linalg.norm(profile) + 1e-9))
    for name, profile in MODE_PROFILES
]


@dataclass
class KeyEstimate:
    key: str
    mode: str
    confidence: float


def _key_from_chroma(chroma_vec: np.ndarray, constrain_root: int | None = None) -> KeyEstimate:
    """Estimate key/mode from a chroma vector.

    If *constrain_root* is provided (a KEYS index), only modes rooted on that
    pitch class are considered.  This is used when the caller has established a
    reliable global root (e.g. from the full-track mean chroma) and wants to
    find the best *mode* for a shorter window without letting a chord transient
    on a different root win the comparison.
    """
    if np.allclose(chroma_vec.sum(), 0.0):
        return KeyEstimate(key="C", mode="major", confidence=0.0)

    chroma_norm = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-9)

    best_score = -1.0
    best_key_idx = 0
    best_mode = "major"

    for mode_name, profile_norm in _MODE_PROFILES_NORM:
        shifts = [constrain_root] if constrain_root is not None else range(12)
        for shift in shifts:
            score = float(np.dot(chroma_norm, np.roll(profile_norm, shift)))
            if score > best_score:
                best_score = score
                best_key_idx = shift
                best_mode = mode_name

    # Locrian is theoretically valid but vanishingly rare as a real key centre;
    # treat it as minor to avoid spurious classifications.
    if best_mode == "locrian":
        best_mode = "minor"

    # The Krumhansl correlation cannot reliably distinguish minor from dorian
    # because the two modes share 6 of their 7 scale degrees.  The only
    # distinguishing note is the 6th: major (dorian) vs. lowered (minor).
    # When the correlation picks minor, compare the chroma energy at the major
    # 6th (+9 semitones from root) against the b6 (+8 semitones).  If the
    # major 6th is clearly stronger the mode is dorian, not minor.
    if best_mode == "minor":
        m6_idx = (best_key_idx + 9) % 12
        b6_idx = (best_key_idx + 8) % 12
        if chroma_vec[m6_idx] > chroma_vec[b6_idx] * 1.1:
            best_mode = "dorian"

    # Similarly, major and mixolydian share 6 of 7 scale degrees; the b7 vs
    # major-7 energy decides which fits.  In EDM the b7 is the more common
    # borrowed degree so an ambiguous major result often belongs to mixolydian.
    if best_mode == "major":
        b7_idx = (best_key_idx + 10) % 12
        maj7_idx = (best_key_idx + 11) % 12
        if chroma_vec[b7_idx] > chroma_vec[maj7_idx] * 1.1:
            best_mode = "mixolydian"

    return KeyEstimate(key=KEYS[best_key_idx], mode=best_mode, confidence=best_score)


def _raw_tempo_windows(beat_times: np.ndarray) -> list[dict]:
    """Per-window BPM estimates before any coalescing."""
    if beat_times.size < 8:
        return []

    raw_segments: list[dict] = []
    win = 16
    hop = 8
    for start in range(0, max(1, beat_times.size - win), hop):
        end_idx = min(start + win, beat_times.size - 1)
        bt = beat_times[start : end_idx + 1]
        ibi = np.diff(bt)
        if ibi.size < 2:
            continue
        bpm = float(60.0 / np.median(ibi))
        confidence = float(np.clip(1.0 - np.std(ibi) / (np.mean(ibi) + 1e-9), 0.0, 1.0))
        raw_segments.append(
            {
                "start": float(bt[0]),
                "end": float(bt[-1]),
                "bpm": bpm,
                "confidence": confidence,
            }
        )
    return raw_segments


def _dedup_tempo_windows(segments: list[dict]) -> list[dict]:
    """Merge consecutive windows with the same integer-rounded BPM for display purposes."""
    if not segments:
        return []
    merged: list[dict] = []
    for seg in segments:
        if not merged:
            merged.append(seg.copy())
            continue
        prev = merged[-1]
        if round(seg["bpm"]) == round(prev["bpm"]):
            prev["end"] = seg["end"]
            prev["bpm"] = float((prev["bpm"] + seg["bpm"]) / 2.0)
            prev["confidence"] = float((prev["confidence"] + seg["confidence"]) / 2.0)
        else:
            merged.append(seg.copy())
    return merged


def _local_tempo_segments(beat_times: np.ndarray) -> list[dict]:
    segs = _coalesce_tempo_segments(_raw_tempo_windows(beat_times), settings.tempo_section_min_delta_bpm)
    return _drop_short_tempo_segments(segs, settings.min_section_tempo_sec, settings.tempo_section_min_delta_bpm)


def _coalesce_tempo_segments(segments: list[dict], min_delta_bpm: float) -> list[dict]:
    if not segments:
        return []

    # Run to fixed-point: a single greedy pass updates rolling BPM averages, which
    # can leave adjacent pairs whose final BPMs are below the threshold (e.g. A+B
    # average drops close enough to C that a second pass would merge them).  Repeat
    # until no further merges occur so that all surviving boundaries are genuinely
    # above the threshold before they are used to derive section edges.
    while True:
        merged: list[dict] = []
        for seg in segments:
            if not merged:
                merged.append(seg.copy())
                continue
            prev = merged[-1]
            if abs(seg["bpm"] - prev["bpm"]) < min_delta_bpm:
                prev["end"] = seg["end"]
                prev["bpm"] = float((prev["bpm"] + seg["bpm"]) / 2.0)
                prev["confidence"] = float((prev["confidence"] + seg["confidence"]) / 2.0)
            else:
                # Overlapping windows mean the previous segment's end may extend
                # past the new segment's start.  Trim to avoid overlap so that
                # downstream section builders see clean, non-overlapping ranges.
                if seg["start"] < prev["end"]:
                    prev["end"] = seg["start"]
                merged.append(seg.copy())
        if len(merged) == len(segments):
            break
        segments = merged

    return merged


def _drop_short_tempo_segments(segments: list[dict], min_sec: float, min_delta_bpm: float) -> list[dict]:
    """Remove tempo segments shorter than min_sec that are sandwiched between
    same-BPM neighbors.

    Hop-grid quantization in the beat tracker can produce a brief phantom
    segment whose BPM differs from both neighbors by more than min_delta_bpm
    (so normal coalescing doesn't absorb it) while those neighbors agree with
    each other (within min_delta_bpm).  A computer-produced track at a fixed
    tempo is the canonical example: two consecutive 16-beat windows straddle a
    region where the median IBI rounds to one frame fewer, yielding e.g. 136 BPM
    between two long 132.51 BPM segments.  Such artifacts are always shorter than
    a musical phrase (< 30 s) and always have same-BPM neighbors.

    Runs to convergence in case multiple adjacent short segments exist.
    """
    if len(segments) <= 1:
        return segments

    segments = [s.copy() for s in segments]
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(segments):
            if seg["end"] - seg["start"] >= min_sec:
                continue
            left = segments[i - 1] if i > 0 else None
            right = segments[i + 1] if i + 1 < len(segments) else None
            if left is None or right is None:
                continue
            if abs(left["bpm"] - right["bpm"]) >= min_delta_bpm:
                continue  # neighbors don't agree — keep the segment
            # Merge: extend left to cover the short segment and the right,
            # then drop both the short segment and the right.
            left["end"] = right["end"]
            left["bpm"] = float((left["bpm"] + right["bpm"]) / 2.0)
            left["confidence"] = float((left["confidence"] + right["confidence"]) / 2.0)
            segments.pop(i + 1)  # drop right
            segments.pop(i)      # drop short
            changed = True
            break

    return segments


def _segment_key_timeline_raw(
    chroma_sync: np.ndarray,
    beat_times: np.ndarray,
    global_root_idx: int | None = None,
    global_root_margin: float = 0.015,
) -> list[dict]:
    """Compute per-window key estimates.

    *global_root_idx* — when provided, each window is scored twice: once
    unrestricted and once with the root constrained to *global_root_idx*.  If
    the unconstrained winner's root differs from *global_root_idx* AND the
    constrained score is within *global_root_margin* of the unconstrained score,
    the constrained (global-root) estimate wins.  This prevents a transient
    chord on a non-tonic root from overriding a stable global key centre that
    the full-track chroma has identified.
    """
    if chroma_sync.shape[1] < 8 or beat_times.size < 8:
        return []

    raw: list[dict] = []
    win = 16
    hop = 8
    for start in range(0, max(1, chroma_sync.shape[1] - win), hop):
        end = min(start + win, chroma_sync.shape[1])
        vec = np.mean(chroma_sync[:, start:end], axis=1)
        est = _key_from_chroma(vec)
        if global_root_idx is not None and KEYS.index(est.key) != global_root_idx:
            constrained = _key_from_chroma(vec, constrain_root=global_root_idx)
            if est.confidence - constrained.confidence <= global_root_margin:
                est = constrained
        start_time = float(beat_times[min(start, beat_times.size - 1)])
        end_time = float(beat_times[min(end - 1, beat_times.size - 1)])
        raw.append(
            {
                "start": start_time,
                "end": end_time,
                "key": est.key,
                "mode": est.mode,
                "confidence": float(np.clip(est.confidence, 0.0, 1.0)),
            }
        )
    return raw


def _coalesce_key_segments_same_label(segments: list[dict]) -> list[dict]:
    if not segments:
        return []
    merged: list[dict] = []
    for seg in segments:
        if not merged:
            merged.append(seg.copy())
            continue
        prev = merged[-1]
        if prev["key"] == seg["key"] and prev["mode"] == seg["mode"]:
            prev["end"] = seg["end"]
            prev["confidence"] = float((prev["confidence"] + seg["confidence"]) / 2)
        else:
            merged.append(seg.copy())
    return merged


def _dominant_key_label(segments: list[dict]) -> tuple[str, str, float]:
    if not segments:
        return "C", "major", 0.0
    weighted = Counter()
    conf_sum = 0.0
    for seg in segments:
        duration = max(1e-6, float(seg["end"]) - float(seg["start"]))
        weight = float(seg["confidence"]) * duration
        weighted[(seg["key"], seg["mode"])] += weight
        conf_sum += weight
    (key, mode), score = weighted.most_common(1)[0]
    return key, mode, float(score / (conf_sum + 1e-9))


def _confirm_key_segments(raw_segments: list[dict]) -> list[dict]:
    if not raw_segments:
        return []

    persist = max(1, settings.key_change_min_persist_windows)
    conf_margin = settings.key_change_conf_margin
    min_conf = settings.key_change_min_confidence

    dominant_key, dominant_mode, dominant_share = _dominant_key_label(raw_segments)
    first_key = raw_segments[0]["key"]
    first_mode = raw_segments[0]["mode"]
    first_run = []
    idx = 0
    while idx < len(raw_segments):
        seg = raw_segments[idx]
        if seg["key"] != first_key or seg["mode"] != first_mode:
            break
        first_run.append(seg)
        idx += 1

    first_run_len = len(first_run)
    first_run_conf = float(np.mean([x["confidence"] for x in first_run])) if first_run else 0.0
    keep_first_as_anchor = first_run_len >= persist and first_run_conf >= min_conf
    anchor_key = first_key if keep_first_as_anchor else dominant_key
    anchor_mode = first_mode if keep_first_as_anchor else dominant_mode
    anchor_conf = first_run_conf if keep_first_as_anchor else float(max(min_conf, dominant_share))

    confirmed: list[dict] = [
        {
            "start": raw_segments[0]["start"],
            "end": raw_segments[0]["end"],
            "key": anchor_key,
            "mode": anchor_mode,
            "confidence": anchor_conf,
        }
    ]
    i = 1

    while i < len(raw_segments):
        current = confirmed[-1]
        seg = raw_segments[i]
        same_label = current["key"] == seg["key"] and current["mode"] == seg["mode"]
        if same_label:
            current["end"] = seg["end"]
            current["confidence"] = float((current["confidence"] + seg["confidence"]) / 2.0)
            i += 1
            continue

        j = i
        run: list[dict] = []
        while j < len(raw_segments):
            cand = raw_segments[j]
            if cand["key"] != seg["key"] or cand["mode"] != seg["mode"]:
                break
            run.append(cand)
            j += 1

        run_len = len(run)
        run_conf = float(np.mean([x["confidence"] for x in run])) if run else 0.0

        passes_persist = run_len >= persist
        passes_conf = run_conf >= max(min_conf, current["confidence"] + conf_margin)

        if passes_persist and passes_conf:
            new_seg = {
                "start": run[0]["start"],
                "end": run[-1]["end"],
                "key": run[0]["key"],
                "mode": run[0]["mode"],
                "confidence": run_conf,
            }
            confirmed.append(new_seg)
        else:
            current["end"] = run[-1]["end"]
            current["confidence"] = float((current["confidence"] + run_conf) / 2.0)

        i = j

    return _coalesce_key_segments_same_label(confirmed)


def _apply_segment_dorian_refinement(
    segments: list[dict],
    chroma_sync: np.ndarray,
    beat_times: np.ndarray,
    threshold: float = 1.02,
) -> list[dict]:
    """Reclassify 'minor' segments to 'dorian' using aggregate chroma evidence.

    Per-window chroma is noisy — a single transient window with m6 > b6 can
    incorrectly flip consecutive-window runs, preventing `_confirm_key_segments`
    from forming a clean minor segment.  Instead this function computes mean
    chroma across all beats *within* each confirmed segment and applies the
    minor→dorian check at that coarser, more stable scale.

    The m6/b6 ratio thresholds from empirical testing:
      - flute (B root):   Ab/G  = 1.041  → dorian  ✓
      - juno  (Eb root):  C/B   = 0.818  → stays minor ✓
      - funky (Ab root):  F/E   = 0.953  → stays minor ✓
    """
    for seg in segments:
        if seg.get("mode") != "minor":
            continue
        key_idx = KEYS.index(seg["key"])
        # Collect beat indices within the segment time range.
        mask = (beat_times >= seg["start"]) & (beat_times <= seg["end"])
        beat_cols = np.where(mask)[0]
        if beat_cols.size == 0:
            continue
        seg_chroma = np.mean(chroma_sync[:, beat_cols], axis=1)
        m6_idx = (key_idx + 9) % 12
        b6_idx = (key_idx + 8) % 12
        if seg_chroma[m6_idx] > seg_chroma[b6_idx] * threshold:
            seg["mode"] = "dorian"
    return segments


def _segment_key_timeline(
    chroma_sync: np.ndarray,
    beat_times: np.ndarray,
    global_tuning_cents: int = 0,
) -> tuple[list[dict], list[dict]]:
    # Derive a global root hint from the full-track mean chroma.
    #
    # Strategy: prefer the Krumhansl correlation winner when it is nearly as
    # energy-strong as the raw argmax (ratio ≥ 0.95).  Krumhansl is mode-aware
    # and correctly handles cases where a prominent b3 bass note has higher raw
    # energy than the tonic (e.g. a Bb-minor track with a heavy Db bass inflating
    # the Db chroma bin above Bb).
    #
    # When the raw argmax has clearly dominant energy (ratio < 0.95), it is a
    # reliable indicator of the true bass root and takes priority.  This handles
    # tracks like A mixolydian where the bass A is so strong that Krumhansl is
    # fooled into picking Gb/F# minor (A is the b3 of Gb in the Krumhansl minor
    # profile, getting a high weight that outscores A mixolydian).
    global_chroma = np.mean(chroma_sync, axis=1)
    argmax_idx = int(np.argmax(global_chroma))
    global_key_est = _key_from_chroma(global_chroma)
    krumhansl_idx = KEYS.index(global_key_est.key)
    if global_chroma[krumhansl_idx] >= global_chroma[argmax_idx] * 0.95:
        global_root_idx = krumhansl_idx
    else:
        global_root_idx = argmax_idx

    _p5_corrected = False  # set to True if the P5-above correction fires below

    # "Detected root is the 5th" correction for minor keys near standard pitch.
    #
    # When Krumhansl and argmax agree on a minor root X but the b2 of X carries
    # more chroma energy than the natural 2 of X, the track may be in a key
    # where X is the dominant (5th) rather than the tonic.  The b2-of-X
    # fingerprint works because the b2 of X equals the b6 of the candidate
    # (X + 5), and the b6 is diatonic in minor while the natural 2 of X is not
    # — so chroma[b2] > chroma[nat2] at the detected root is strong evidence
    # the track is in the P4-above key.
    #
    # Tuning gate: large tuning offsets cause the tonic to spill energy into the
    # adjacent semitone bin (the b2), triggering false positives.  Only apply
    # the correction when tuning is within ±20 cents of standard pitch.
    #
    # Confirmed on: Alpaca (G→C minor, margin 1.4%) and
    #               Deflection (G→C minor, margin 0.8%).
    if (
        global_root_idx == krumhansl_idx
        and global_key_est.mode in {"minor", "locrian"}
        and abs(global_tuning_cents) < 20
    ):
        _b2_idx = (global_root_idx + 1) % 12
        _nat2_idx = (global_root_idx + 2) % 12
        if global_chroma[_b2_idx] > global_chroma[_nat2_idx] * 1.25:
            _cand_root = (global_root_idx + 5) % 12
            _chroma_n = global_chroma / (np.linalg.norm(global_chroma) + 1e-9)
            _minor_norm = next(p for n, p in _MODE_PROFILES_NORM if n == "minor")
            _cand_score = float(np.dot(_chroma_n, np.roll(_minor_norm, _cand_root)))
            _curr_score = float(np.dot(_chroma_n, np.roll(_minor_norm, global_root_idx)))
            if _cand_score > _curr_score * 0.97:
                global_root_idx = _cand_root
                _p5_corrected = True

    # When the P5-above correction fires the per-window constrained call must
    # always override the unconstrained estimate, so raise the margin to 1.0
    # (confidence is bounded to [0, 1], so this forces the corrected root).
    _window_margin = 1.0 if _p5_corrected else 0.015

    raw = _segment_key_timeline_raw(
        chroma_sync, beat_times,
        global_root_idx=global_root_idx,
        global_root_margin=_window_margin,
    )
    # When the P5-above correction fires, the constrained call forces root to
    # the corrected pitch class but the mode may come out as lydian (because
    # the Krumhansl lydian profile gives its highest weight to the *original*
    # dominant note, which is now being treated as the 5th of the corrected
    # root).  Override all windows on the corrected root to "minor" so that
    # downstream dorian refinement has the right base mode to work from.
    if _p5_corrected:
        _cand_key = KEYS[global_root_idx]
        for _seg in raw:
            if _seg["key"] == _cand_key and _seg["mode"] != "minor":
                _seg["mode"] = "minor"
    raw_coalesced = _coalesce_key_segments_same_label(raw)
    confirmed = _confirm_key_segments(raw)
    # Fallback: if _confirm_key_segments collapses the track to a single segment
    # (common when a track modulates through several modes within each key centre,
    # so no (key, mode) pair runs long enough to confirm), try root-level
    # detection.  This groups windows by root only, ignoring mode cycling, and
    # determines mode from aggregate chroma per confirmed root period.
    if len(confirmed) <= 1:
        root_confirmed = _confirm_root_change_segments(raw, chroma_sync, beat_times)
        if len(root_confirmed) >= 2:
            confirmed = root_confirmed
            raw_coalesced = root_confirmed
    # Refine minor→dorian using aggregate chroma within each segment.
    # Per-window detection is too noisy for this distinction; segment-mean
    # chroma gives a reliable m6/b6 ratio (see _apply_segment_dorian_refinement).
    confirmed = _apply_segment_dorian_refinement(confirmed, chroma_sync, beat_times)
    raw_coalesced = _apply_segment_dorian_refinement(raw_coalesced, chroma_sync, beat_times)
    return raw_coalesced, confirmed


def _confirm_root_change_segments(
    raw_segments: list[dict],
    chroma_sync: np.ndarray,
    beat_times: np.ndarray,
    bin_sec: float = 60.0,
    min_run_sec: float = 90.0,
) -> list[dict]:
    """Detect structural one-way key changes using large-window chroma analysis.

    Divides the track into time bins and computes the dominant root of each bin
    from its *aggregate* chroma (unconstrained Krumhansl correlation).  Aggregate
    chroma over a large window is far more stable than per-window estimates:
    the tonal centre repeats throughout and accumulates dominant energy, while
    diatonically-related chord tones (IV, V, VI…) only appear transiently.

    This handles tracks like Jettison (Ab drone intro → D major section) where
    the D section has many per-window estimates of G, Gb, A (all diatonic to D
    major) that fragment a per-window neighbourhood vote, but whose aggregate
    chroma clearly points to D.

    Consecutive bins with the same root are grouped into runs; runs shorter than
    min_run_sec are discarded (filters brief transition bins).  A cycling-
    modulation guard then rejects tracks where the same root appears in more than
    one run (e.g. B → Ab → B → Ab), as these represent oscillating tonality
    rather than a structural one-way modulation.

    For each confirmed root run, mode is determined from its aggregate chroma
    (constrained to the confirmed root).

    Returns ≥2 segments when a genuine structural key change is detected,
    otherwise [].  Used as a fallback in _segment_key_timeline when
    _confirm_key_segments produces only a single segment.
    """
    if beat_times.size < 2:
        return []
    duration = float(beat_times[-1])
    if duration < min_run_sec * 2:
        return []

    # Aggregate chroma per time bin → unconstrained root estimate.
    bin_edges = list(np.arange(0.0, duration, bin_sec)) + [duration]
    bin_roots: list[tuple[float, float, str, float]] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (beat_times >= lo) & (beat_times < hi)
        cols = np.where(mask)[0]
        if cols.size < 4:
            continue
        agg = np.mean(chroma_sync[:, cols], axis=1)
        est = _key_from_chroma(agg)
        bin_roots.append((lo, hi, est.key, float(est.confidence)))

    if len(bin_roots) < 2:
        return []

    # Group consecutive bins with the same root into runs.
    runs: list[dict] = []
    i = 0
    while i < len(bin_roots):
        root = bin_roots[i][2]
        j = i + 1
        while j < len(bin_roots) and bin_roots[j][2] == root:
            j += 1
        start_t = bin_roots[i][0]
        end_t = bin_roots[j - 1][1]
        if end_t - start_t >= min_run_sec:
            root_idx = KEYS.index(root)
            mask = (beat_times >= start_t) & (beat_times <= end_t)
            cols = np.where(mask)[0]
            if cols.size > 0:
                agg = np.mean(chroma_sync[:, cols], axis=1)
                mode_est = _key_from_chroma(agg, constrain_root=root_idx)
                mode = mode_est.mode
            else:
                mode = bin_roots[i][2]
            runs.append({
                "start": start_t,
                "end": end_t,
                "key": root,
                "mode": mode,
                "confidence": float(np.mean([b[3] for b in bin_roots[i:j]])),
            })
        i = j

    if len(runs) < 2:
        return []

    # Reject cycling modulations: a structural key change introduces each root
    # only once.  If any root appears in more than one run the track is cycling
    # between tonalities (e.g. B → Ab → B → Ab) and the fallback should not
    # activate.
    root_seq = [r["key"] for r in runs]
    if len(set(root_seq)) < len(root_seq):
        return []

    # Extend boundary runs to cover the full track; fill gaps between runs by
    # splitting at the midpoint between adjacent confirmed ends/starts.
    runs[0]["start"] = 0.0
    runs[-1]["end"] = duration
    for k in range(1, len(runs)):
        mid = (runs[k - 1]["end"] + runs[k]["start"]) / 2.0
        runs[k - 1]["end"] = mid
        runs[k]["start"] = mid

    return runs


def _global_key_from_segments(key_segments: list[dict]) -> dict:
    if not key_segments:
        return {"key": "C", "mode": "major", "confidence": 0.0}
    weighted = Counter()
    total_conf = 0.0
    for seg in key_segments:
        label = f"{seg['key']}:{seg['mode']}"
        weighted[label] += seg["confidence"]
        total_conf += seg["confidence"]
    top, score = weighted.most_common(1)[0]
    key, mode = top.split(":", 1)
    return {
        "key": key,
        "mode": mode,
        "confidence": float(score / (total_conf + 1e-9)),
    }


def _dominant_key_in_interval(segments: list[dict], start: float, end: float) -> dict:
    if not segments:
        return {"key": None, "mode": None, "confidence": 0.0}
    weighted = Counter()
    total = 0.0
    for seg in segments:
        overlap_start = max(start, float(seg["start"]))
        overlap_end = min(end, float(seg["end"]))
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue
        weight = overlap * float(seg.get("confidence", 0.0))
        weighted[(seg.get("key"), seg.get("mode"))] += weight
        total += weight

    if not weighted:
        mid = (start + end) / 2.0
        fallback = _value_at_time(segments, mid, ["key", "mode", "confidence"])
        return {
            "key": fallback.get("key"),
            "mode": fallback.get("mode"),
            "confidence": float(fallback.get("confidence") or 0.0),
        }

    (key, mode), score = weighted.most_common(1)[0]
    return {"key": key, "mode": mode, "confidence": float(score / (total + 1e-9))}


def _dominant_tempo_in_interval(segments: list[dict], start: float, end: float) -> dict:
    if not segments:
        return {"bpm": None, "confidence": 0.0}
    weighted_bpm = 0.0
    total_weight = 0.0
    for seg in segments:
        overlap_start = max(start, float(seg["start"]))
        overlap_end = min(end, float(seg["end"]))
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue
        conf = float(seg.get("confidence", 0.0))
        weight = overlap * max(conf, 1e-6)
        weighted_bpm += float(seg.get("bpm", 0.0)) * weight
        total_weight += weight

    if total_weight <= 0:
        mid = (start + end) / 2.0
        fallback = _value_at_time(segments, mid, ["bpm", "confidence"])
        return {
            "bpm": fallback.get("bpm"),
            "confidence": float(fallback.get("confidence") or 0.0),
        }

    return {"bpm": float(weighted_bpm / total_weight), "confidence": float(min(1.0, total_weight))}


def _value_at_time(segments: list[dict], t: float, value_keys: list[str]) -> dict:
    if not segments:
        return {k: None for k in value_keys}
    for seg in segments:
        if float(seg["start"]) <= t <= float(seg["end"]):
            return {k: seg.get(k) for k in value_keys}
    last = segments[-1]
    return {k: last.get(k) for k in value_keys}


def _build_sections(
    tempo_segments: list[dict],
    key_segments: list[dict],
    key_value_segments: list[dict],
    duration_sec: float,
    fuzz_sec: float,
) -> list[dict]:
    events: list[dict] = []

    for i in range(1, len(tempo_segments)):
        prev = tempo_segments[i - 1]
        cur = tempo_segments[i]
        events.append(
            {
                "time": float(cur["start"]),
                "tempo_change": True,
                "key_change": False,
                "from_bpm": prev.get("bpm"),
                "to_bpm": cur.get("bpm"),
                "from_key": None,
                "from_mode": None,
                "to_key": None,
                "to_mode": None,
            }
        )

    for i in range(1, len(key_segments)):
        prev = key_segments[i - 1]
        cur = key_segments[i]
        events.append(
            {
                "time": float(cur["start"]),
                "tempo_change": False,
                "key_change": True,
                "from_bpm": None,
                "to_bpm": None,
                "from_key": prev.get("key"),
                "from_mode": prev.get("mode"),
                "to_key": cur.get("key"),
                "to_mode": cur.get("mode"),
            }
        )

    if not events:
        bpm_info = _dominant_tempo_in_interval(tempo_segments, 0.0, float(duration_sec))
        key_info = _dominant_key_in_interval(key_value_segments, 0.0, float(duration_sec))
        return [
            {
                "start": 0.0,
                "end": float(duration_sec),
                "tempo_bpm": bpm_info["bpm"],
                "key": key_info["key"],
                "mode": key_info["mode"],
                "starts_with_tempo_change": False,
                "starts_with_key_change": False,
                "change_reasons": [],
            }
        ]

    events.sort(key=lambda x: x["time"])
    clusters: list[list[dict]] = [[events[0]]]
    for event in events[1:]:
        last_cluster = clusters[-1]
        cluster_time = float(np.mean([e["time"] for e in last_cluster]))
        if abs(event["time"] - cluster_time) <= fuzz_sec:
            last_cluster.append(event)
        else:
            clusters.append([event])

    merged_events: list[dict] = []
    for cluster in clusters:
        t = float(np.mean([e["time"] for e in cluster]))
        tempo_changes = [e for e in cluster if e["tempo_change"]]
        key_changes = [e for e in cluster if e["key_change"]]
        tempo_event = tempo_changes[-1] if tempo_changes else None
        key_event = key_changes[-1] if key_changes else None
        merged_events.append(
            {
                "time": t,
                "tempo_change": bool(tempo_event),
                "key_change": bool(key_event),
                "from_bpm": tempo_event.get("from_bpm") if tempo_event else None,
                "to_bpm": tempo_event.get("to_bpm") if tempo_event else None,
                "from_key": key_event.get("from_key") if key_event else None,
                "from_mode": key_event.get("from_mode") if key_event else None,
                "to_key": key_event.get("to_key") if key_event else None,
                "to_mode": key_event.get("to_mode") if key_event else None,
            }
        )

    boundaries = [0.0]
    boundary_events: dict[float, dict] = {}
    for event in merged_events:
        t = float(np.clip(event["time"], 0.0, max(0.0, duration_sec)))
        if t <= boundaries[-1] + 1e-4:
            t = boundaries[-1] + 1e-4
        if t >= duration_sec:
            continue
        boundaries.append(t)
        boundary_events[t] = event
    boundaries.append(float(duration_sec))

    overall: list[dict] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        bpm_info = _dominant_tempo_in_interval(tempo_segments, start, end)
        key_info = _dominant_key_in_interval(key_value_segments, start, end)
        boundary_event = boundary_events.get(start)
        starts_with_tempo_change = bool(boundary_event["tempo_change"]) if boundary_event else False
        starts_with_key_change = bool(boundary_event["key_change"]) if boundary_event else False
        reasons = []
        if starts_with_tempo_change:
            reasons.append("tempo")
        if starts_with_key_change:
            reasons.append("key")

        overall.append(
            {
                "start": float(start),
                "end": float(end),
                "tempo_bpm": bpm_info["bpm"],
                "key": key_info.get("key"),
                "mode": key_info.get("mode"),
                "starts_with_tempo_change": starts_with_tempo_change,
                "starts_with_key_change": starts_with_key_change,
                "change_reasons": reasons,
                "change_detail": boundary_event if boundary_event else None,
            }
        )
    return overall


def _tuning_cents_from_offset(offset_semitones: float) -> int:
    cents = int(round(float(offset_semitones) * 100.0))
    return max(-49, min(50, cents))


def _apply_section_tuning(
    sections: list[dict],
    y: np.ndarray | None = None,
    sr: int | None = None,
    global_tuning_cents: int = 0,
) -> list[dict]:
    """Attach tuning fields to each section.

    If *y* and *sr* are provided the global and per-section tuning are estimated
    from the audio (original behaviour).  When only *global_tuning_cents* is
    supplied (interpret-only path) that value is applied to every section.
    """
    if not sections:
        return sections

    if y is not None and sr is not None:
        try:
            global_offset = float(librosa.estimate_tuning(y=y, sr=sr, n_fft=4096))
        except Exception:  # noqa: BLE001
            global_offset = 0.0
        global_tuning_cents = _tuning_cents_from_offset(global_offset)

    tuned: list[dict] = []
    for seg in sections:
        cents = global_tuning_cents
        if y is not None and sr is not None:
            start_t = float(seg["start"])
            end_t = float(seg["end"])
            s = max(0, int(start_t * sr))
            e = min(len(y), int(end_t * sr))
            if e - s >= max(2048, sr // 2):
                y_seg = y[s:e]
                try:
                    seg_offset = float(librosa.estimate_tuning(y=y_seg, sr=sr, n_fft=4096))
                    cents = _tuning_cents_from_offset(seg_offset)
                except Exception:  # noqa: BLE001
                    pass
        item = seg.copy()
        item["tuning"] = int(cents)
        item["tuning_rounded"] = int(round(cents / 25) * 25)
        tuned.append(item)
    return tuned


def _coalesce_sections_by_content(sections: list[dict], min_tempo_delta_bpm: float) -> list[dict]:
    if not sections:
        return []

    merged: list[dict] = []
    for sec in sections:
        if not merged:
            merged.append(sec.copy())
            continue

        prev = merged[-1]
        same_key = prev.get("key") == sec.get("key") and prev.get("mode") == sec.get("mode")
        prev_bpm = prev.get("tempo_bpm")
        curr_bpm = sec.get("tempo_bpm")
        if prev_bpm is None or curr_bpm is None:
            tempo_close = True
        else:
            tempo_close = abs(float(prev_bpm) - float(curr_bpm)) < float(min_tempo_delta_bpm)

        if same_key and tempo_close:
            prev["end"] = sec["end"]
            if prev_bpm is not None and curr_bpm is not None:
                prev["tempo_bpm"] = float((float(prev_bpm) + float(curr_bpm)) / 2.0)
            prev["starts_with_tempo_change"] = bool(prev.get("starts_with_tempo_change", False))
            prev["starts_with_key_change"] = bool(prev.get("starts_with_key_change", False))
            prev["change_reasons"] = prev.get("change_reasons", [])
            prev["change_detail"] = prev.get("change_detail")
        else:
            merged.append(sec.copy())

    return merged


def _tonal_family(mode: str) -> str:
    if mode in _MINOR_FAMILY:
        return "minor"
    if mode in _MAJOR_FAMILY:
        return "major"
    return mode


def _coalesce_same_root_mode_family(sections: list[dict]) -> list[dict]:
    """Merge adjacent sections that share the same root and tonal family.

    A track is extremely unlikely to "modulate" between minor modes (e.g.
    minor → dorian) or between major modes (e.g. major → mixolydian) on the
    same root — these are Krumhansl correlation noise, not real key changes.
    Merge such adjacent pairs into one section using the mode of the
    longer-lasting one.  BPM is duration-weighted across the merge.

    Sections with significantly different BPMs are kept separate even if the
    key family matches — a genuine tempo change (e.g. ritardando) should not be
    collapsed into a blended BPM.
    """
    if not sections:
        return []
    merged: list[dict] = [sections[0].copy()]
    for sec in sections[1:]:
        prev = merged[-1]
        bpm_prev = prev.get("tempo_bpm")
        bpm_cur = sec.get("tempo_bpm")
        if bpm_prev is not None and bpm_cur is not None:
            tempo_close = abs(float(bpm_prev) - float(bpm_cur)) < settings.tempo_section_min_delta_bpm
        else:
            tempo_close = True
        if (
            prev.get("key") == sec.get("key")
            and _tonal_family(prev.get("mode", "")) == _tonal_family(sec.get("mode", ""))
            and _tonal_family(prev.get("mode", "")) in ("minor", "major")
            and tempo_close
        ):
            dur_prev = prev["end"] - prev["start"]
            dur_cur = sec["end"] - sec["start"]
            # Keep the mode of the longer section.
            if dur_cur > dur_prev:
                prev["mode"] = sec["mode"]
            # Duration-weighted BPM.
            if bpm_prev is not None and bpm_cur is not None:
                prev["tempo_bpm"] = (bpm_prev * dur_prev + bpm_cur * dur_cur) / (dur_prev + dur_cur)
            prev["end"] = sec["end"]
        else:
            merged.append(sec.copy())
    return merged


def _snap_sections_to_dominant_tempo(sections: list[dict], max_spread_bpm: float = 5.0) -> list[dict]:
    """If all section BPMs fall within max_spread_bpm, snap them to the dominant.

    Tracks produced on a sequencer have a single stable tempo; minor estimation
    noise (e.g. arising from beat-grid quantisation or a tempo correction factor)
    should not produce a spread of reported BPMs.  When the full spread is below
    the threshold, all sections are snapped to the duration-weighted dominant so
    that bpm and bpm_end always agree.
    """
    if not sections:
        return sections
    bpms = [s["tempo_bpm"] for s in sections if s.get("tempo_bpm") is not None]
    if len(bpms) < 2 or max(bpms) - min(bpms) >= max_spread_bpm:
        return sections
    # Use the minimum BPM rather than the weighted average: estimation errors are
    # systematically positive (mean IBI inflated by hop-grid quantisation; tempo
    # correction factor may overshoot), so the lowest section BPM is the best
    # approximation of the true tempo.
    dominant = min(bpms)
    result = []
    for s in sections:
        s = s.copy()
        if s.get("tempo_bpm") is not None:
            s["tempo_bpm"] = dominant
        result.append(s)
    return result


def _key_pitch_classes(key: str, mode: str) -> frozenset:
    """Return the set of absolute pitch classes for a given key/mode."""
    _MODE_INTERVALS: dict[str, list[int]] = {
        "major":      [0, 2, 4, 5, 7, 9, 11],
        "minor":      [0, 2, 3, 5, 7, 8, 10],
        "dorian":     [0, 2, 3, 5, 7, 9, 10],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "phrygian":   [0, 1, 3, 5, 7, 8, 10],
        "locrian":    [0, 1, 3, 5, 6, 8, 10],
        "lydian":     [0, 2, 4, 6, 7, 9, 11],
    }
    root = KEYS.index(key) if key in KEYS else 0
    intervals = _MODE_INTERVALS.get(mode, _MODE_INTERVALS["major"])
    return frozenset((root + i) % 12 for i in intervals)


def _drop_short_key_sections(sections: list[dict], min_sec: float = 30.0) -> list[dict]:
    """Absorb key sections shorter than min_sec into their harmonically closest neighbor.

    In EDM a section lasting under 30 seconds is almost always an intro/outro
    artifact or detection noise rather than a genuine modulation.  The short
    section is merged into whichever adjacent section (a) is at least min_sec
    long, and (b) shares the most pitch classes with the short section.  When
    both neighbors qualify and are tied on pitch-class overlap, the longer
    neighbor wins.

    Runs to convergence so that stacked short intro sections are fully
    collapsed in one call.
    """
    if len(sections) <= 1:
        return sections

    sections = [s.copy() for s in sections]
    changed = True
    while changed:
        changed = False
        for i, sec in enumerate(sections):
            if sec["end"] - sec["start"] >= min_sec:
                continue

            left = sections[i - 1] if i > 0 else None
            right = sections[i + 1] if i + 1 < len(sections) else None
            left_dur = (left["end"] - left["start"]) if left else 0.0
            right_dur = (right["end"] - right["start"]) if right else 0.0
            left_ok = left is not None and left_dur >= min_sec
            right_ok = right is not None and right_dur >= min_sec

            if not left_ok and not right_ok:
                continue  # no eligible neighbor — leave it

            sec_pcs = _key_pitch_classes(sec.get("key", "C"), sec.get("mode", "major"))
            left_shared = len(sec_pcs & _key_pitch_classes(left["key"], left["mode"])) if left_ok else -1
            right_shared = len(sec_pcs & _key_pitch_classes(right["key"], right["mode"])) if right_ok else -1

            absorb_left = left_ok and (
                not right_ok
                or left_shared > right_shared
                or (left_shared == right_shared and left_dur >= right_dur)
            )

            if absorb_left:
                left["end"] = sec["end"]
                # Refresh boundary flags on the section now adjacent to the extended left.
                if right is not None:
                    _refresh_boundary(right, left)
            else:
                right["start"] = sec["start"]
                _refresh_boundary(right, left)

            sections.pop(i)
            changed = True
            break  # restart scan with updated list

    return sections


def _refresh_boundary(section: dict, prev: dict | None) -> None:
    """Recalculate starts_with_* flags on *section* given its new predecessor."""
    if prev is None:
        section["starts_with_key_change"] = False
        section["starts_with_tempo_change"] = False
        section["change_reasons"] = []
        return
    section["starts_with_key_change"] = (
        prev.get("key") != section.get("key")
        or prev.get("mode") != section.get("mode")
    )
    p_bpm = prev.get("tempo_bpm")
    s_bpm = section.get("tempo_bpm")
    section["starts_with_tempo_change"] = (
        p_bpm is not None and s_bpm is not None
        and abs(float(p_bpm) - float(s_bpm)) >= 2.0
    )
    reasons = []
    if section["starts_with_key_change"]:
        reasons.append("key")
    if section["starts_with_tempo_change"]:
        reasons.append("tempo")
    section["change_reasons"] = reasons


def _build_form_sections(duration: float, rms: np.ndarray, sr: int, hop_length: int) -> list[dict]:
    if rms.size == 0:
        return []
    n_sections = max(3, min(8, int(duration // 45) + 2))
    boundaries = np.linspace(0, duration, n_sections + 1)
    sections: list[dict] = []
    for i in range(n_sections):
        start = float(boundaries[i])
        end = float(boundaries[i + 1])
        mid = (start + end) / 2.0
        start_f = int(start * sr / hop_length)
        end_f = int(end * sr / hop_length)
        energy = float(np.mean(rms[start_f : max(start_f + 1, end_f)])) if end_f > start_f else 0.0

        if i == 0:
            label = "intro"
        elif i == n_sections - 1:
            label = "outro"
        elif energy > np.percentile(rms, 75):
            label = "drop"
        elif mid < duration * 0.4:
            label = "build"
        else:
            label = "breakdown"

        sections.append({"start": start, "end": end, "label": label, "confidence": 0.55})
    return sections


def _beat_attack_sustain_ratios(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
) -> list[float]:
    """Return attack/sustain RMS ratio for each beat using a peak-relative split.

    The boundary between attack and sustain is the sample of maximum absolute
    amplitude within the beat.  For percussion the peak is within the first few
    milliseconds, so the sustain window captures the full ring/decay.  For a pad
    or fading-in melody the peak arrives late or spans the beat, pushing the ratio
    toward or below 1.  High ratio = percussive; ratio ≈ 1 or lower = sustained/harmonic.
    """
    ratios = []
    for i in range(beat_times.size - 1):
        start = max(0, int(beat_times[i] * sr))
        end = min(len(y), int(beat_times[i + 1] * sr))
        beat = y[start:end]
        if beat.size < 4:
            ratios.append(1.0)
            continue
        peak_idx = int(np.argmax(np.abs(beat)))
        # Require at least 1 sample on each side so the split is meaningful.
        peak_idx = max(1, min(peak_idx, beat.size - 2))
        attack_rms = float(np.sqrt(np.mean(beat[:peak_idx] ** 2)))
        sustain_rms = float(np.sqrt(np.mean(beat[peak_idx:] ** 2)))
        ratios.append(attack_rms / (sustain_rms + 1e-9))
    return ratios


def _find_harmonic_start(
    beat_times: np.ndarray,
    chroma_sync: np.ndarray,
    percussive_ratio_per_frame: np.ndarray,
    beat_attack_sustain: list[float],
    sr: int,
    hop_length: int,
    threshold: float,
    consecutive: int,
    perc_threshold: float,
    atk_threshold: float,
) -> tuple[float, float, float, int, list, list]:
    if beat_times.size < 2 or chroma_sync.shape[1] == 0:
        return 0.0, 0.0, 0.0, 0, [], []

    # Per-beat tonal confidence from key correlation.
    tonal_conf = []
    for i in range(chroma_sync.shape[1]):
        tonal_conf.append(_key_from_chroma(chroma_sync[:, i]).confidence)
    tonal_conf_arr = np.array(tonal_conf, dtype=float)

    # Per-beat mean percussive ratio (same aggregation as _non_harmonic_segments).
    per_beat_perc = []
    for i in range(min(chroma_sync.shape[1], beat_times.size - 1)):
        s = int(beat_times[i] * sr / hop_length)
        e = int(beat_times[i + 1] * sr / hop_length)
        if e <= s:
            e = s + 1
        per_beat_perc.append(float(np.mean(percussive_ratio_per_frame[s:e])))
    while len(per_beat_perc) < len(tonal_conf):
        per_beat_perc.append(per_beat_perc[-1] if per_beat_perc else 1.0)
    perc_arr = np.array(per_beat_perc, dtype=float)

    # Per-beat attack/sustain ratio (pre-computed in phase 1).
    atk_list = list(beat_attack_sustain)
    while len(atk_list) < len(tonal_conf):
        atk_list.append(atk_list[-1] if atk_list else 1.0)
    atk_arr = np.array(atk_list[:len(tonal_conf)], dtype=float)

    def _scan(require_perc: bool, max_atk: float) -> int:
        """Return start beat index of first harmonic window, or -1 if none found.

        Pass 1 (require_perc=True): both perc_ratio AND attack/sustain must be
        below their thresholds — handles tracks where HPSS clearly separates
        percussion from harmony.

        Pass 2 (require_perc=False): only attack/sustain is checked — fallback
        for tracks with tonal/ringy percussion that fools HPSS (low perc_ratio
        throughout).  Uses a stricter atk threshold to avoid triggering on
        transitional windows that mix percussion and harmonic beats.

        Both passes also require chroma movement: a window where all beats are
        dominated by the same pitch class is a bass drone or tonal percussion,
        not a melodic entry.  We require at least 3 distinct dominant pitch
        classes across the consecutive beats.  This threshold is calibrated for
        the default consecutive=8; re-evaluate if that setting changes.
        """
        for i in range(0, max(1, tonal_conf_arr.size - consecutive + 1)):
            tonal_window = tonal_conf_arr[i : i + consecutive]
            perc_window = perc_arr[i : i + consecutive]
            atk_window = atk_arr[i : i + consecutive]
            tonal_ok = np.all(tonal_window >= threshold)
            perc_ok = np.mean(perc_window) < perc_threshold
            # Require both mean AND peak atk below thresholds.  A window where
            # even one beat has a very high atk (drum fill, crash) is still a
            # transitional zone and should not count as the harmonic entry.
            atk_ok = np.mean(atk_window) < max_atk and np.max(atk_window) < max_atk * 2
            n_cols = chroma_sync.shape[1]
            chroma_window = chroma_sync[:, i : min(i + consecutive, n_cols)]
            dom = np.argmax(chroma_window, axis=0)
            chroma_ok = len(set(dom.tolist())) >= 3
            if tonal_ok and atk_ok and chroma_ok and (perc_ok or not require_perc):
                return i
        return -1

    idx_pass1 = _scan(require_perc=True, max_atk=atk_threshold)
    # Pass 2 uses a stricter atk ceiling (half of the pass-1 value) so that
    # transitional windows where some beats still have percussion transients
    # don't trigger a false early detection.
    idx_pass2 = _scan(require_perc=False, max_atk=atk_threshold * 0.5)
    # Pass 2 is only informative when the opening window had elevated atk,
    # i.e. a clear percussion-only intro existed before the harmonic entry.
    # If atk is already uniformly low from beat 0 (e.g. a track with quiet,
    # low-atk percussion throughout), pass 2 fires at the first beat with
    # chroma variety regardless of the true harmonic entry — a false positive.
    # In that case pass 1 (perc-ratio gate) is the correct signal; discard pass 2.
    opening_atk_mean = float(np.mean(atk_arr[:consecutive]))
    pass2_meaningful = opening_atk_mean >= atk_threshold * 0.5
    # Pass 0: check whether harmonic content is already present at beat 0,
    # i.e. drums and melody coexist from the very start with no drum-only intro.
    # We use the MEDIAN (not mean) of the opening window's atk values so that
    # sparse drum transients (e.g. a kick every 4 beats) don't inflate the
    # aggregate and hide the otherwise-low-atk harmonic beats.
    # A pure drum intro will have uniformly high atk across all beats in the
    # window, keeping the median well above atk_threshold.
    #
    # Secondary condition: if ANY single beat in the opening window has an
    # extremely low atk (< atk_threshold × 0.25), that beat almost certainly
    # carries a pure sustained tone (synth pad, held chord) rather than a
    # percussion hit — percussion cannot produce such a near-zero attack value.
    # This catches dense breakbeat tracks where the median atk is elevated by
    # drum transients but one beat clearly belongs to a harmonic element.
    # Pass 0 also requires that the opening window has chroma movement — same
    # ≥3 distinct dominant pitches guard used in _scan.  A bass drone that plays
    # throughout a drum intro creates strong tonal_conf and low atk but keeps
    # the chroma locked on a single pitch class; we must not misclassify that
    # as "melody present from beat 0."
    opening_dom = np.argmax(chroma_sync[:, : min(consecutive, chroma_sync.shape[1])], axis=0)
    opening_chroma_varied = len(set(opening_dom.tolist())) >= 3
    if (
        tonal_conf_arr.size >= consecutive
        and np.all(tonal_conf_arr[:consecutive] >= threshold)
        and opening_chroma_varied
    ):
        opening_atk = atk_arr[:consecutive]
        if np.median(opening_atk) < atk_threshold or np.min(opening_atk) < atk_threshold * 0.25:
            idx_pass0 = 0
        else:
            idx_pass0 = -1
    else:
        idx_pass0 = -1
    # Take the earliest non-negative result across all passes.  Pass 1 can find
    # a "clean" harmonic moment later in the track (e.g. a drum breakdown) that
    # is technically valid but is not the actual melody entry.  Earlier passes
    # represent better candidates for the true harmonic start.
    active_passes = [idx_pass0, idx_pass1]
    if pass2_meaningful:
        active_passes.append(idx_pass2)
    candidates = [i for i in active_passes if i >= 0]

    # Pre-beat check: if the beat tracker started well after t=0 (e.g. because
    # the track opens with softly-attacked pad synths that produce weak onsets),
    # check whether the pre-beat audio is low-percussive.  If so, harmonic
    # content was present before the first detected beat, meaning there is no
    # percussion-only intro → clamp start_idx to 0.
    # Guard: only applies when beat_times[0] is more than 2 beat-lengths after
    # t=0 to avoid triggering on the normal beat-tracker startup latency.
    if beat_times.size >= 2:
        beat_period = float(np.median(np.diff(beat_times)))
        pre_beat_sec = float(beat_times[0])
        pre_beat_frames = int(pre_beat_sec * sr / hop_length)
        min_gap_sec = beat_period * 2
        if pre_beat_sec > min_gap_sec and pre_beat_frames > 0:
            pre_perc = float(np.mean(percussive_ratio_per_frame[:pre_beat_frames]))
            if pre_perc < perc_threshold:
                candidates = [0] + candidates

    idx = min(candidates) if candidates else 0
    start_idx = max(0, idx)

    start_time = float(beat_times[min(start_idx, beat_times.size - 1)])
    start_conf = float(np.mean(tonal_conf_arr[start_idx : start_idx + consecutive]))

    beats_before = max(0, start_idx)
    bars_4_4 = float(beats_before / 4.0)
    return bars_4_4, round(bars_4_4), start_time, start_conf, tonal_conf_arr.tolist(), perc_arr.tolist()


def _non_harmonic_segments(
    beat_times: np.ndarray,
    chroma_sync: np.ndarray,
    percussive_ratio_per_frame: np.ndarray,
    hop_length: int,
    sr: int,
) -> list[dict]:
    if beat_times.size < 2 or chroma_sync.shape[1] == 0:
        return []

    per_beat_perc = []
    for i in range(min(chroma_sync.shape[1], beat_times.size - 1)):
        s = int(beat_times[i] * sr / hop_length)
        e = int(beat_times[i + 1] * sr / hop_length)
        if e <= s:
            e = s + 1
        per_beat_perc.append(float(np.mean(percussive_ratio_per_frame[s:e])))

    segments = []
    run_start = None
    for i in range(min(len(per_beat_perc), chroma_sync.shape[1])):
        tonal = _key_from_chroma(chroma_sync[:, i]).confidence
        percussive = per_beat_perc[i]
        non_harmonic = tonal < 0.26 and percussive > 0.55
        if non_harmonic and run_start is None:
            run_start = i
        if (not non_harmonic or i == len(per_beat_perc) - 1) and run_start is not None:
            end_idx = i if not non_harmonic else i + 1
            if end_idx - run_start >= 2:
                segments.append(
                    {
                        "start": float(beat_times[run_start]),
                        "end": float(beat_times[min(end_idx, beat_times.size - 1)]),
                        "reason": "low_tonal_high_percussive",
                        "confidence": 0.6,
                    }
                )
            run_start = None
    return segments


def _percussion_presence(
    onset_env: np.ndarray,
    y_harm: np.ndarray | None = None,
    y_perc: np.ndarray | None = None,
    sr: int = 44100,
    perc_energy_ratio: float | None = None,
    perc_ratio_p95: float | None = None,
    beat_atk_p95: float | None = None,
) -> tuple[str, bool, float]:
    onset_p95 = float(np.percentile(onset_env, 95))
    if perc_energy_ratio is not None:
        perc_ratio = perc_energy_ratio
    else:
        perc_ratio = float(np.sum(np.abs(y_perc)) / (np.sum(np.abs(y_harm)) + np.sum(np.abs(y_perc)) + 1e-9))

    score = 0.5 * min(1.0, perc_ratio / 0.7) + 0.5 * min(1.0, onset_p95 / 2.0)
    if score < 0.18:
        level = "none"
    elif score < 0.35:
        level = "low"
    elif score < 0.7:
        level = "normal"
    else:
        level = "high"

    low = level in {"none", "low"}

    # Secondary no-drums gate: check whether any frame is *dominantly* percussive.
    # Real drum hits push individual HPSS frames well above 0.5 percussive ratio;
    # staccato synth attacks are brief and spectrally narrow, so even the 95th
    # percentile of per-frame percussive ratio stays near the floor.
    # Empirical separation: jettison (no drums, staccato synth) p95=0.501;
    # all drums tracks p95 ≥ 0.620. Threshold at 0.55 gives clear headroom.
    #
    # Two rescue paths bypass this gate when HPSS separation is unreliable:
    # 1. High-attack rescue: very large per-beat attack/sustain ratios mean real
    #    drum hits whose broadband energy confused HPSS.
    # 2. Tonal-kick rescue: extremely low perc_ratio_p95 (< 0.35) combined with
    #    a high primary score means resonant/tonal kicks that HPSS routed to the
    #    harmonic component — the contradiction signals HPSS failure, not silence.
    if perc_ratio_p95 is not None:
        _high_atk = beat_atk_p95 is not None and beat_atk_p95 >= settings.perc_hpss_rescue_atk_p95
        _tonal_kick = perc_ratio_p95 < 0.35 and score > 0.55
        if not (_high_atk or _tonal_kick) and perc_ratio_p95 < 0.55:
            low = True
            if level not in {"none", "low"}:
                level = "low"

    confidence = float(np.clip(abs(score - 0.35) + 0.4, 0.0, 1.0))
    return level, low, confidence


def _swing_from_grid(beat_times: np.ndarray, onset_times: np.ndarray) -> tuple[bool, float]:
    if beat_times.size < 8 or onset_times.size < 8:
        return False, 0.0

    ratios = []
    for i in range(beat_times.size - 1):
        left = beat_times[i]
        right = beat_times[i + 1]
        dur = right - left
        if dur <= 1e-5:
            continue
        mid = left + 0.5 * dur
        window_left = left + 0.3 * dur
        window_right = left + 0.8 * dur
        candidates = onset_times[(onset_times >= window_left) & (onset_times <= window_right)]
        if candidates.size == 0:
            continue
        candidate = candidates[np.argmin(np.abs(candidates - mid))]
        ratio = float((candidate - left) / dur)
        ratios.append(ratio)

    if len(ratios) < 12:
        return False, 0.0

    r = np.array(ratios)
    median = float(np.median(r))
    spread = float(np.std(r))

    # Score A — consistent late offbeat (jazz / house swing): the median
    # mid-beat onset falls noticeably after the straight 0.5 position.
    # Stability penalises high spread because randomly scattered onsets
    # can accidentally produce a late median.
    raw_score = max(0.0, (median - 0.53) / 0.17)
    stability = max(0.0, 1.0 - (spread / 0.12))
    median_score = raw_score * stability

    # Score B — sparse late offbeat (breakbeat / funk swing): most beats
    # play nearly straight, but a significant fraction (~10-18 %) has a
    # clearly late note (ratio > 0.65).  Straight tracks have essentially
    # zero such beats.  This captures breakbeat swing that does not raise
    # the median appreciably because the majority of sub-beat onsets are
    # still close to 0.5.
    #
    # Two false-positive guards:
    #   • late_frac > 0.25: too many late onsets to be breakbeat swing —
    #     this is a regular subdivision (e.g. 16th-note hi-hat landing at
    #     75% of the beat on almost every beat).
    #   • late_std < 0.015: the late onsets are mechanically quantised to a
    #     fixed grid position (tightly clustered ≈ 0.75); genuine swing
    #     timing always has more beat-to-beat variation (std ≥ 0.018).
    late_frac = float(np.mean(r > 0.65))
    _late_r = r[r > 0.65]
    _late_std = float(np.std(_late_r)) if _late_r.size > 1 else 0.0
    _late_ok = late_frac <= 0.25 and _late_std >= 0.015
    late_score = max(0.0, (late_frac - 0.04) / 0.08) if _late_ok else 0.0

    score = float(np.clip(max(median_score, late_score), 0.0, 1.0))
    return bool(score > 0.45), score


def _detect_tempo_correction_ratio(
    onset_env: np.ndarray,
    beat_times: np.ndarray,
    sr: int,
    hop_length: int,
    tg_ratio_threshold: float = 0.75,
    tg_ratio_threshold_32: float = 0.85,
) -> float:
    """Return a correction multiplier when the beat tracker locked onto the wrong tempo.

    Two error modes are detected:

    ×0.75 (4/3 downward): beat tracker locked onto 4/3 × the true tempo.
    The tempogram shows a strong peak at detected BPM and a nearly-as-strong
    peak at 3/4 × detected (the true tempo).  Fires when the 3/4 energy is
    at least `tg_ratio_threshold` of the detected energy.

    ×1.5 (3/2 upward): beat tracker locked onto 2/3 × the true tempo.
    The tempogram shows a strong peak at detected BPM and a nearly-as-strong
    peak at 3/2 × detected (the true tempo).  Fires when the 3/2 energy is
    at least `tg_ratio_threshold_32` of the detected energy.

    Returns 1.0 (no correction) for all other cases.
    """
    if beat_times.size < 8:
        return 1.0
    detected_bpm = 60.0 / float(np.median(np.diff(beat_times)))

    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tempo_axis = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=hop_length)
    mean_tg = np.mean(tg, axis=1)
    bw = 2.0

    def _bpm_energy(target: float) -> float:
        mask = np.abs(tempo_axis - target) < bw
        return float(np.mean(mean_tg[mask])) if mask.any() else 0.0

    e_det = _bpm_energy(detected_bpm)
    if e_det <= 0:
        return 1.0

    # Check 4/3 downward correction (detected is too fast).
    # Suppress if the track shows a 3/2 ambiguity signature: when the tempogram
    # has nearly equal energy at 2/3×detected (the true tempo for a 3/2-error
    # track), the candidate at 0.75×detected is a false positive caused by
    # broad tempogram peaks rather than a genuine 4/3 tracking error.
    e_34 = _bpm_energy(detected_bpm * 0.75)
    e_23 = _bpm_energy(detected_bpm * 0.667)
    if e_34 / e_det >= tg_ratio_threshold and e_23 / e_det < tg_ratio_threshold_32:
        return 0.75

    # Check 3/2 upward correction (detected is too slow)
    e_32 = _bpm_energy(detected_bpm * 1.5)
    if e_32 / e_det >= tg_ratio_threshold_32:
        return 1.5

    return 1.0


def extract_audio_features(path: str) -> dict:
    """Phase 1: load audio and run all expensive librosa computations.

    Returns a JSON-serializable dict of arrays that can be cached and passed
    to interpret_features() without re-loading the audio file.
    """
    y, sr = librosa.load(path, sr=settings.analysis_sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Detect and correct 4/3 (downward) and 3/2 (upward) tempo tracking errors
    # at extraction time so that chroma_sync is aligned to the true beat grid.
    # After correction, detected_bpm≈true and neither candidate fires again,
    # so double-correction is naturally prevented.
    if beat_times.size >= 8:
        _extraction_corr = _detect_tempo_correction_ratio(onset_env, beat_times, sr, hop_length)
        if _extraction_corr != 1.0:
            # Use the BPM implied by beat_times (not tempo_raw) so that both
            # upward (×1.5) and downward (×0.75) corrections target the right BPM.
            # tightness=300 (vs the default 100) forces the tracker to commit to the
            # corrected tempo prior rather than drifting back to the wrong grid.
            _detected_bpm = 60.0 / float(np.median(np.diff(beat_times)))
            _corrected_start_bpm = _detected_bpm * _extraction_corr
            _, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, hop_length=hop_length,
                start_bpm=_corrected_start_bpm,
            )
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="time")

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.mean) if beat_frames.size > 1 else chroma

    y_harm, y_perc = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_perc = librosa.feature.rms(y=y_perc, hop_length=hop_length)[0]
    rms_harm = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]
    percussive_ratio_per_frame = rms_perc / (rms_harm + rms_perc + 1e-9)

    beat_attack_sustain = _beat_attack_sustain_ratios(y, sr, beat_times)

    try:
        global_tuning_cents = _tuning_cents_from_offset(float(librosa.estimate_tuning(y=y, sr=sr, n_fft=4096)))
    except Exception:  # noqa: BLE001
        global_tuning_cents = 0

    perc_energy_ratio = float(
        np.sum(np.abs(y_perc)) / (np.sum(np.abs(y_harm)) + np.sum(np.abs(y_perc)) + 1e-9)
    )

    return {
        "sr": int(sr),
        "hop_length": hop_length,
        "duration": duration,
        "tempo_raw": float(tempo),
        "beat_times": beat_times.tolist(),
        "onset_times": onset_times.tolist(),
        "onset_env": onset_env.tolist(),
        "chroma_sync": chroma_sync.tolist(),
        "percussive_ratio_per_frame": percussive_ratio_per_frame.tolist(),
        "beat_attack_sustain": beat_attack_sustain,
        "rms": rms.tolist(),
        "global_tuning_cents": global_tuning_cents,
        "perc_energy_ratio": perc_energy_ratio,
    }


def save_features(features: dict, path: str) -> None:
    """Persist phase-1 features to disk as JSON."""
    import json
    from pathlib import Path
    Path(path).write_text(json.dumps(features))


def load_features(path: str) -> dict:
    """Load phase-1 features previously saved by save_features()."""
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text())


def interpret_features(features: dict, profile: str = "edm_v1") -> dict:
    """Phase 2: derive musical interpretation from pre-computed audio features.

    This is the fast path — no audio file needed.  Changing thresholds in
    app/config.py and re-running this function is sufficient for most tuning work.
    """
    start = datetime.utcnow()

    sr = features["sr"]
    hop_length = features["hop_length"]
    duration = features["duration"]
    tempo_raw = features["tempo_raw"]
    beat_times = np.array(features["beat_times"])
    onset_times = np.array(features["onset_times"])
    onset_env = np.array(features["onset_env"])
    chroma_sync = np.array(features["chroma_sync"])
    percussive_ratio_per_frame = np.array(features["percussive_ratio_per_frame"])
    beat_attack_sustain = features["beat_attack_sustain"]
    rms = np.array(features["rms"])
    global_tuning_cents = features["global_tuning_cents"]
    perc_energy_ratio = features["perc_energy_ratio"]

    key_segments_raw, key_segments = _segment_key_timeline(
        chroma_sync, beat_times, global_tuning_cents=global_tuning_cents
    )
    global_key = _global_key_from_segments(key_segments_raw if key_segments_raw else key_segments)

    raw_tempo_segments = _dedup_tempo_windows(_raw_tempo_windows(beat_times))
    tempo_segments = _local_tempo_segments(beat_times)
    # Recompute each segment's BPM as the mean IBI of the clean hop-grid beats.
    # The median-based window BPMs used during coalescing are robust for segment
    # boundary detection, but when the beat tracker quantises positions to the
    # STFT hop grid the median systematically picks the shorter IBI in a bimodal
    # (alternating short/long) distribution, biasing BPM upward by ~1.  The mean
    # of clean IBIs (within ±1.5 hops of the median) corrects this: the two
    # neighbouring hop-grid values differ by exactly one hop so both are included,
    # while spurious short IBIs from false beat detections are excluded.
    _hop_sec = hop_length / sr
    for _seg in tempo_segments:
        _seg_beats = beat_times[(beat_times >= _seg["start"]) & (beat_times <= _seg["end"])]
        if _seg_beats.size >= 2:
            _ibis = np.diff(_seg_beats)
            _med = float(np.median(_ibis))
            _clean = _ibis[np.abs(_ibis - _med) <= 1.5 * _hop_sec]
            if _clean.size >= 2:
                _seg["bpm"] = float(60.0 / np.mean(_clean))

    # Detect and correct 4/3 / 3/2 tempo tracking error per segment.
    #
    # A global correction (applied uniformly to all segments) breaks tracks
    # with a genuine tempo change where only one section has a 4/3 error.
    # Example: a track at 133 BPM that ritardandos to 102 BPM will have the
    # beat tracker lock onto 136 = 4/3×102 for the second half; the global
    # tempogram then shows e(102)/e(136) ≈ 0.85, firing the 4/3 correction
    # and wrongly rescaling the 133-BPM first section to ~99 BPM.
    #
    # Per-segment correction: compute the tempogram once, then for each
    # segment independently check whether its BPM has a strong 3/4
    # sub-harmonic.  The first section (132.51 BPM) scores e(99)/e(133) ≈
    # 0.74 — just below threshold — while the second (136 BPM) scores 0.85
    # and is correctly rescaled to 102.
    #
    # _tempo_correction_bars tracks whether the segment that contains the
    # harmonic start was corrected, so bars_4_4 can be scaled accordingly.
    # Use the global tempogram (averaged over the full track) and gate each
    # segment's correction on whether its BPM is within min_delta_bpm of the
    # globally detected median tempo.  Segments that belong to a genuinely
    # different tempo region (e.g. the 133 BPM first half of a track that
    # ritardandos to 102 while the beat tracker locks onto 136 = 4/3×102 for
    # the second half) are left uncorrected; only the segments tracking the
    # same grid error as the detected BPM receive the correction.
    _detected_bpm_global = 60.0 / float(np.median(np.diff(beat_times)))
    _tempo_correction = _detect_tempo_correction_ratio(
        onset_env, beat_times, sr, hop_length,
        tg_ratio_threshold=settings.tempo_correction_tg_ratio,
    )
    if _tempo_correction != 1.0:
        for _seg in tempo_segments:
            if abs(_seg["bpm"] - _detected_bpm_global) < settings.tempo_section_min_delta_bpm:
                _seg["bpm"] *= _tempo_correction
        for _seg in raw_tempo_segments:
            if abs(_seg["bpm"] - _detected_bpm_global) < settings.tempo_section_min_delta_bpm:
                _seg["bpm"] *= _tempo_correction

    sections = _build_sections(
        tempo_segments=tempo_segments,
        key_segments=key_segments,
        key_value_segments=key_segments_raw if key_segments_raw else key_segments,
        duration_sec=duration,
        fuzz_sec=settings.segment_boundary_fuzz_sec,
    )
    sections = _coalesce_sections_by_content(sections, settings.tempo_section_min_delta_bpm)
    sections = _coalesce_same_root_mode_family(sections)
    sections = _snap_sections_to_dominant_tempo(sections)
    sections = _drop_short_key_sections(sections, min_sec=settings.min_section_key_sec)
    sections = _apply_section_tuning(sections, global_tuning_cents=global_tuning_cents)
    for s in sections:
        s["tempo_bpm_rounded"] = int(round(s["tempo_bpm"])) if s.get("tempo_bpm") is not None else None

    global_bpm = float(np.median([s["bpm"] for s in tempo_segments])) if tempo_segments else float(tempo_raw)
    global_bpm_conf = (
        float(np.mean([s["confidence"] for s in tempo_segments])) if tempo_segments else 0.35
    )

    bars_4_4, _, _, _, beat_tonal_conf, beat_perc_ratio = _find_harmonic_start(
        beat_times,
        chroma_sync,
        percussive_ratio_per_frame,
        beat_attack_sustain,
        sr,
        hop_length,
        settings.harmonic_conf_threshold,
        settings.harmonic_consecutive_beats,
        settings.harmonic_perc_threshold,
        settings.harmonic_atk_threshold,
    )

    if _tempo_correction != 1.0:
        bars_4_4 *= _tempo_correction

    perc_ratio_p95 = float(np.percentile(percussive_ratio_per_frame, 95))
    beat_atk_p95 = float(np.percentile(beat_attack_sustain, 95))
    percussion_presence, low_percussion, percussion_conf = _percussion_presence(
        onset_env,
        perc_energy_ratio=perc_energy_ratio,
        perc_ratio_p95=perc_ratio_p95,
        beat_atk_p95=beat_atk_p95,
    )
    _ = percussion_presence, percussion_conf

    swing_feel, swing_confidence = _swing_from_grid(beat_times, onset_times)
    _ = swing_confidence

    form_sections = _build_form_sections(duration, rms, sr, hop_length)

    elapsed = (datetime.utcnow() - start).total_seconds()

    # A no-drums track has no percussion-only intro by definition.
    if low_percussion:
        bars_4_4 = 0.0

    # Detect tracks with no reliable tempo: when there are no drums AND the onset
    # envelope is very smooth (low coefficient of variation), the beat tracker has
    # nothing to lock onto and produces artefact beats from synth attacks or reverb
    # tails.  Suppress section BPMs and swing so callers get None rather than a
    # spurious fast tempo (e.g. 198 BPM for a droning ambient pad track).
    _onset_cv = float(np.std(onset_env) / (np.mean(onset_env) + 1e-8))
    no_tempo = bool(low_percussion and _onset_cv < settings.tempo_reliable_onset_cv_min)
    if no_tempo:
        for s in sections:
            s["tempo_bpm"] = None
            s["tempo_bpm_rounded"] = None
        swing_feel = False
        # Tempo boundaries are meaningless when BPM is unreliable.  Collapse
        # consecutive sections that share the same key+mode into one so the
        # caller sees a clean single section rather than many spurious splits.
        collapsed: list[dict] = []
        for s in sections:
            if (
                collapsed
                and collapsed[-1].get("key") == s.get("key")
                and collapsed[-1].get("mode") == s.get("mode")
            ):
                collapsed[-1]["end"] = s["end"]
                collapsed[-1]["change_reasons"] = [
                    r for r in collapsed[-1]["change_reasons"] if r != "tempo"
                ]
                collapsed[-1]["starts_with_tempo_change"] = False
            else:
                collapsed.append(s)
        sections = collapsed

    # Detect atonal / no-key tracks: count pitch classes that win more than 2%
    # of beat windows.  Real harmonic tracks have 5+ competitive PCs; tracks
    # whose chroma is driven entirely by tuned percussion (e.g. 808 toms) have
    # ≤4 because one or two fixed pitches dominate every window.
    #
    # Exempt no_tempo tracks from this check: an ambient pad with only 2-3
    # sustained notes (e.g. F and C droning) will have few competitive PCs but
    # is genuinely harmonic.  The low-PC condition only arises from tuned
    # percussion in rhythmic (non-ambient) contexts.
    _dominant_pc = chroma_sync.argmax(axis=0)
    _pc_fracs = np.bincount(_dominant_pc, minlength=12) / max(_dominant_pc.size, 1)
    _n_competitive_pcs = int((_pc_fracs > 0.02).sum())
    # Guard against drum-contaminated chroma: when drums produce broadband
    # transients, all 12 pitch classes are elevated nearly equally and the
    # argmax is decided by noise rather than tonal content.  In such frames
    # the winner's margin over the runner-up is tiny (~0.05) whereas genuine
    # pitch concentration (e.g. tuned 808 toms) produces a large margin
    # (~0.27+).  Only count the competitive-PC result as meaningful when the
    # mean winner margin is above the threshold.
    _chroma_sorted = np.sort(chroma_sync, axis=0)[::-1, :]
    _winner_margin = float(np.mean(_chroma_sorted[0, :] - _chroma_sorted[1, :]))
    no_key = (
        (not no_tempo)
        and (_n_competitive_pcs <= settings.no_key_max_competitive_pcs)
        and (_winner_margin >= settings.no_key_min_winner_margin)
    )
    if no_key:
        for s in sections:
            s["key"] = None
            s["mode"] = None
            s["tuning_cents"] = None
            s["tuning_rounded"] = None

    return {
        "global": {
            "swing": bool(swing_feel),
            "no_drums": bool(low_percussion),
            "no_tempo": bool(no_tempo),
            "no_key": bool(no_key),
            "bars_percussion": float(bars_4_4),
            "bars_percussion_rounded": (
                0 if bars_4_4 < 0.9 else
                min([0,1,2,3,4]+[8*k for k in range(1,50)], key=lambda x: abs(x-bars_4_4))
            ),
        },
        "sections": sections,
        "debug": {
            "raw_tempo_segments": raw_tempo_segments,
            "tempo_segments": tempo_segments,
            "raw_key_segments": key_segments_raw,
            "beat_tonal_conf": beat_tonal_conf,
            "beat_perc_ratio": beat_perc_ratio,
            "beat_attack_sustain": beat_attack_sustain,
            "collapsed_key_segments": [
                {
                    **seg,
                    **_dominant_key_in_interval(
                        key_segments_raw if key_segments_raw else key_segments,
                        float(seg["start"]),
                        float(seg["end"]),
                    ),
                }
                for seg in key_segments
            ],
        },
        "form_sections": form_sections,
        "provenance": {
            "analysis_profile": profile,
            "algorithm_versions": {
                "tempo": "librosa.beat_track",
                "key": "krumhansl_correlation",
                "swing": "offbeat_delay_v1",
                "percussion_presence": "hpss_onset_hybrid_v1",
            },
            "thresholds": {
                "harmonic_conf_threshold": settings.harmonic_conf_threshold,
                "harmonic_consecutive_beats": settings.harmonic_consecutive_beats,
                "tempo_section_min_delta_bpm": settings.tempo_section_min_delta_bpm,
                "key_change_min_persist_windows": settings.key_change_min_persist_windows,
                "key_change_conf_margin": settings.key_change_conf_margin,
                "key_change_min_confidence": settings.key_change_min_confidence,
                "segment_boundary_fuzz_sec": settings.segment_boundary_fuzz_sec,
            },
            "track_info": {
                "duration_sec": duration,
                "sample_rate": sr,
                "channels": 1,
            },
            "global_estimates_internal": {
                "bpm": global_bpm,
                "key": global_key["key"],
                "mode": global_key["mode"],
                "confidence": float(np.clip((global_bpm_conf + global_key["confidence"]) / 2, 0.0, 1.0)),
            },
            "run_duration_sec": elapsed,
        },
    }


def analyze_audio_file(path: str, profile: str = "edm_v1") -> dict:
    """Extract audio features and interpret them in one pass."""
    return interpret_features(extract_audio_features(path), profile)
