from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import librosa
import numpy as np

from app.config import settings

KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

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
    return _coalesce_tempo_segments(_raw_tempo_windows(beat_times), settings.tempo_section_min_delta_bpm)


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
                merged.append(seg.copy())
        if len(merged) == len(segments):
            break
        segments = merged

    return merged


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


def _segment_key_timeline(chroma_sync: np.ndarray, beat_times: np.ndarray) -> tuple[list[dict], list[dict]]:
    # Derive a global root hint from the full-track mean chroma.  In bass-driven
    # music the tonic is the note with the highest long-term chroma energy (bass
    # drone / repeating root bass hits dominate the chroma accumulation).  Using
    # the peak-energy bin — rather than the Krumhansl correlation winner — avoids
    # having the higher b6 weight in the minor profile override a track whose
    # bassline is on a different root (e.g. A mixolydian, where the harmonic
    # series of the A bass can inflate the b6 energy of Gb/F# minor, causing the
    # correlation to mis-identify the root as Gb).
    global_chroma = np.mean(chroma_sync, axis=1)
    global_root_idx = int(np.argmax(global_chroma))

    raw = _segment_key_timeline_raw(chroma_sync, beat_times, global_root_idx=global_root_idx)
    raw_coalesced = _coalesce_key_segments_same_label(raw)
    confirmed = _confirm_key_segments(raw)
    return raw_coalesced, confirmed


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
            global_offset = float(librosa.estimate_tuning(y=y, sr=sr))
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
                    seg_offset = float(librosa.estimate_tuning(y=y_seg, sr=sr))
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
        """
        for i in range(0, max(1, tonal_conf_arr.size - consecutive + 1)):
            tonal_window = tonal_conf_arr[i : i + consecutive]
            perc_window = perc_arr[i : i + consecutive]
            atk_window = atk_arr[i : i + consecutive]
            tonal_ok = np.all(tonal_window >= threshold)
            perc_ok = np.mean(perc_window) < perc_threshold
            atk_ok = np.mean(atk_window) < max_atk
            if tonal_ok and atk_ok and (perc_ok or not require_perc):
                return i
        return -1

    idx_pass1 = _scan(require_perc=True, max_atk=atk_threshold)
    # Pass 2 uses a stricter atk ceiling (half of the pass-1 value) so that
    # transitional windows where some beats still have percussion transients
    # don't trigger a false early detection.
    idx_pass2 = _scan(require_perc=False, max_atk=atk_threshold * 0.5)
    # Pass 0: check whether harmonic content is already present at beat 0,
    # i.e. drums and melody coexist from the very start with no drum-only intro.
    # We use the MEDIAN (not mean) of the opening window's atk values so that
    # sparse drum transients (e.g. a kick every 4 beats) don't inflate the
    # aggregate and hide the otherwise-low-atk harmonic beats.
    # A pure drum intro will have uniformly high atk across all beats in the
    # window, keeping the median well above atk_threshold.
    if (tonal_conf_arr.size >= consecutive
            and np.all(tonal_conf_arr[:consecutive] >= threshold)
            and np.median(atk_arr[:consecutive]) < atk_threshold):
        idx_pass0 = 0
    else:
        idx_pass0 = -1
    # Take the earliest non-negative result across all passes.  Pass 1 can find
    # a "clean" harmonic moment later in the track (e.g. a drum breakdown) that
    # is technically valid but is not the actual melody entry.  Earlier passes
    # represent better candidates for the true harmonic start.
    candidates = [i for i in [idx_pass0, idx_pass1, idx_pass2] if i >= 0]
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
) -> tuple[str, bool, float]:
    onset_mean = float(np.mean(onset_env))
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

    # Straight ~= 0.5, swung offbeat is typically delayed.
    raw_score = max(0.0, (median - 0.53) / 0.17)
    stability = max(0.0, 1.0 - (spread / 0.12))
    score = float(np.clip(raw_score * stability, 0.0, 1.0))
    return bool(score > 0.45), score


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
        global_tuning_cents = _tuning_cents_from_offset(float(librosa.estimate_tuning(y=y, sr=sr)))
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

    key_segments_raw, key_segments = _segment_key_timeline(chroma_sync, beat_times)
    global_key = _global_key_from_segments(key_segments_raw if key_segments_raw else key_segments)

    raw_tempo_segments = _dedup_tempo_windows(_raw_tempo_windows(beat_times))
    tempo_segments = _local_tempo_segments(beat_times)
    sections = _build_sections(
        tempo_segments=tempo_segments,
        key_segments=key_segments,
        key_value_segments=key_segments_raw if key_segments_raw else key_segments,
        duration_sec=duration,
        fuzz_sec=settings.segment_boundary_fuzz_sec,
    )
    sections = _coalesce_sections_by_content(sections, settings.tempo_section_min_delta_bpm)
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

    percussion_presence, low_percussion, percussion_conf = _percussion_presence(
        onset_env,
        perc_energy_ratio=perc_energy_ratio,
    )
    _ = percussion_presence, percussion_conf

    swing_feel, swing_confidence = _swing_from_grid(beat_times, onset_times)
    _ = swing_confidence

    form_sections = _build_form_sections(duration, rms, sr, hop_length)

    elapsed = (datetime.utcnow() - start).total_seconds()

    return {
        "global": {
            "swing": bool(swing_feel),
            "no_drums": bool(low_percussion),
            "bars_percussion": float(bars_4_4),
            "bars_percussion_rounded": int(round(bars_4_4)),
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
