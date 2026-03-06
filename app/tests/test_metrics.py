from __future__ import annotations

import numpy as np

from app.services.analysis import (
    _build_sections,
    _coalesce_sections_by_content,
    _coalesce_tempo_segments,
    _confirm_key_segments,
    _dominant_key_label,
    _dominant_tempo_in_interval,
    _find_harmonic_start,
    _local_tempo_segments,
    _percussion_presence,
    _swing_from_grid,
    _tuning_cents_from_offset,
)


def test_harmonic_start_bars_count():
    sr = 22050
    hop_length = 512
    beat_times = np.arange(0, 40, dtype=float) * 0.5
    chroma_sync = np.zeros((12, beat_times.size))

    # Harmonic content starts at beat 16 and stays stable.
    chroma_sync[0, 16:] = 1.0

    # Percussive ratio: high (>0.55) for beats 0-15, low for beats 16+.
    n_frames = int(beat_times[-1] * sr / hop_length) + 2
    perc_ratio = np.ones(n_frames, dtype=float) * 0.8
    beat16_frame = int(beat_times[16] * sr / hop_length)
    perc_ratio[beat16_frame:] = 0.1

    bars, rounded, start_time, conf, _, _ = _find_harmonic_start(
        beat_times=beat_times,
        chroma_sync=chroma_sync,
        percussive_ratio_per_frame=perc_ratio,
        sr=sr,
        hop_length=hop_length,
        threshold=0.1,
        consecutive=4,
        perc_threshold=0.40,
    )

    assert bars == 4.0
    assert rounded == 4
    assert start_time == beat_times[16]
    assert conf >= 0.1


def test_swing_detection_straight_vs_swung():
    beat_times = np.arange(0, 64, dtype=float) * 0.5

    straight_onsets = []
    swung_onsets = []
    for i in range(len(beat_times) - 1):
        l = beat_times[i]
        r = beat_times[i + 1]
        dur = r - l
        straight_onsets.append(l + 0.50 * dur)
        swung_onsets.append(l + 0.64 * dur)

    straight, straight_score = _swing_from_grid(beat_times, np.array(straight_onsets))
    swung, swung_score = _swing_from_grid(beat_times, np.array(swung_onsets))

    assert straight is False
    assert swung is True
    assert swung_score > straight_score


def test_low_percussion_detection():
    onset_env = np.zeros(100)
    y_harm = np.ones(1000)
    y_perc = np.zeros(1000)

    level, low, conf = _percussion_presence(onset_env, y_harm, y_perc, 44100)
    assert level in {"none", "low"}
    assert low is True
    assert 0.0 <= conf <= 1.0


def test_tempo_segments_require_2_bpm_change():
    # First half ~120 BPM, then slight drift to ~121 BPM (should merge),
    # then clear jump to ~124 BPM (should become a new section).
    ibis = np.array([0.5] * 20 + [60.0 / 121.0] * 20 + [60.0 / 124.0] * 20, dtype=float)
    beat_times = np.concatenate(([0.0], np.cumsum(ibis)))

    segments = _local_tempo_segments(beat_times)

    assert len(segments) == 2
    assert abs(segments[0]["bpm"] - 120.5) < 1.5
    assert abs(segments[1]["bpm"] - 124.0) < 1.5


def test_coalesce_tempo_segments_converges_after_intermediate_averaging():
    # Single-pass coalescing: A(128)+B(129.5) → avg 128.75; C(131) is 2.25 above
    # that average so it stays as a new segment; C+D(129.5) → avg 130.25.
    # After one pass: [128.75, 130.25] — delta 1.5, still below the 2.0 threshold.
    # A second pass must merge these, confirming fixed-point convergence is needed.
    raw = [
        {"start": 0.0, "end": 5.0, "bpm": 128.0, "confidence": 0.9},
        {"start": 5.0, "end": 10.0, "bpm": 129.5, "confidence": 0.9},
        {"start": 10.0, "end": 15.0, "bpm": 131.0, "confidence": 0.9},
        {"start": 15.0, "end": 20.0, "bpm": 129.5, "confidence": 0.9},
    ]
    merged = _coalesce_tempo_segments(raw, min_delta_bpm=2.0)
    assert len(merged) == 1


def test_coalesce_tempo_segments_for_existing_results():
    raw = [
        {"start": 0.0, "end": 10.0, "bpm": 120.1853, "confidence": 0.9},
        {"start": 10.0, "end": 20.0, "bpm": 120.1853, "confidence": 0.88},
        {"start": 20.0, "end": 30.0, "bpm": 121.2, "confidence": 0.85},
        {"start": 30.0, "end": 40.0, "bpm": 124.4, "confidence": 0.84},
    ]
    merged = _coalesce_tempo_segments(raw, min_delta_bpm=2.0)
    assert len(merged) == 2
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 30.0
    assert merged[1]["start"] == 30.0


def test_key_confirmation_reduces_false_positive_flips():
    raw = [
        {"start": 0, "end": 8, "key": "D#", "mode": "minor", "confidence": 0.58},
        {"start": 8, "end": 16, "key": "D#", "mode": "minor", "confidence": 0.56},
        {"start": 16, "end": 24, "key": "F#", "mode": "major", "confidence": 0.47},  # short noisy flip
        {"start": 24, "end": 32, "key": "D#", "mode": "minor", "confidence": 0.57},
        {"start": 32, "end": 40, "key": "D#", "mode": "minor", "confidence": 0.59},
    ]
    confirmed = _confirm_key_segments(raw)
    assert len(confirmed) == 1
    assert confirmed[0]["key"] == "D#"
    assert confirmed[0]["mode"] == "minor"


def test_key_confirmation_keeps_sustained_real_change():
    raw = [
        {"start": 0, "end": 8, "key": "D#", "mode": "minor", "confidence": 0.58},
        {"start": 8, "end": 16, "key": "D#", "mode": "minor", "confidence": 0.56},
        {"start": 16, "end": 24, "key": "D#", "mode": "minor", "confidence": 0.57},
        {"start": 24, "end": 32, "key": "A", "mode": "minor", "confidence": 0.70},
        {"start": 32, "end": 40, "key": "A", "mode": "minor", "confidence": 0.72},
        {"start": 40, "end": 48, "key": "A", "mode": "minor", "confidence": 0.69},
    ]
    confirmed = _confirm_key_segments(raw)
    assert len(confirmed) == 2
    assert confirmed[0]["key"] == "D#"
    assert confirmed[1]["key"] == "A"


def test_key_confirmation_uses_dominant_anchor_not_first_window():
    raw = [
        {"start": 0, "end": 8, "key": "G", "mode": "major", "confidence": 0.49},   # wrong intro guess
        {"start": 8, "end": 16, "key": "D#", "mode": "minor", "confidence": 0.68},
        {"start": 16, "end": 24, "key": "D#", "mode": "minor", "confidence": 0.71},
        {"start": 24, "end": 32, "key": "D#", "mode": "minor", "confidence": 0.70},
        {"start": 32, "end": 40, "key": "D#", "mode": "minor", "confidence": 0.69},
    ]
    key, mode, _ = _dominant_key_label(raw)
    confirmed = _confirm_key_segments(raw)
    assert key == "D#"
    assert mode == "minor"
    assert confirmed[0]["key"] == "D#"
    assert confirmed[0]["mode"] == "minor"


def test_sections_fuzzy_match_tempo_and_key_boundaries():
    tempo_segments = [
        {"start": 0.0, "end": 60.0, "bpm": 128.0, "confidence": 0.9},
        {"start": 40.20, "end": 120.0, "bpm": 130.0, "confidence": 0.9},
    ]
    key_segments = [
        {"start": 0.0, "end": 60.0, "key": "D#", "mode": "minor", "confidence": 0.7},
        {"start": 40.65, "end": 120.0, "key": "F#", "mode": "minor", "confidence": 0.7},
    ]
    overall = _build_sections(
        tempo_segments=tempo_segments,
        key_segments=key_segments,
        key_value_segments=key_segments,
        duration_sec=120.0,
        fuzz_sec=0.75,
    )

    assert len(overall) == 2
    assert overall[1]["starts_with_tempo_change"] is True
    assert overall[1]["starts_with_key_change"] is True
    assert "tempo" in overall[1]["change_reasons"]
    assert "key" in overall[1]["change_reasons"]


def test_sections_use_raw_key_evidence_not_single_midpoint_guess():
    tempo_segments = [{"start": 0.0, "end": 100.0, "bpm": 128.0, "confidence": 0.9}]
    confirmed_key_segments = [{"start": 0.0, "end": 100.0, "key": "G", "mode": "major", "confidence": 0.51}]
    raw_key_segments = [
        {"start": 0.0, "end": 10.0, "key": "G", "mode": "major", "confidence": 0.55},
        {"start": 10.0, "end": 100.0, "key": "D#", "mode": "minor", "confidence": 0.75},
    ]
    sections = _build_sections(
        tempo_segments=tempo_segments,
        key_segments=confirmed_key_segments,
        key_value_segments=raw_key_segments,
        duration_sec=100.0,
        fuzz_sec=0.75,
    )
    assert len(sections) == 1
    assert sections[0]["key"] == "D#"
    assert sections[0]["mode"] == "minor"


def test_tuning_cents_conversion_and_clamp():
    assert _tuning_cents_from_offset(0.0) == 0
    assert _tuning_cents_from_offset(0.123) == 12
    assert _tuning_cents_from_offset(-0.234) == -23
    assert _tuning_cents_from_offset(0.9) == 50
    assert _tuning_cents_from_offset(-0.9) == -49


def test_section_coalesce_ignores_subthreshold_tempo_drift():
    sections = [
        {
            "start": 0.0,
            "end": 30.0,
            "tempo_bpm": 120.1,
            "key": "D#",
            "mode": "minor",
            "starts_with_tempo_change": False,
            "starts_with_key_change": False,
            "change_reasons": [],
        },
        {
            "start": 30.0,
            "end": 60.0,
            "tempo_bpm": 121.6,
            "key": "D#",
            "mode": "minor",
            "starts_with_tempo_change": True,
            "starts_with_key_change": False,
            "change_reasons": ["tempo"],
        },
        {
            "start": 60.0,
            "end": 90.0,
            "tempo_bpm": 124.8,
            "key": "D#",
            "mode": "minor",
            "starts_with_tempo_change": True,
            "starts_with_key_change": False,
            "change_reasons": ["tempo"],
        },
    ]
    merged = _coalesce_sections_by_content(sections, min_tempo_delta_bpm=2.0)
    assert len(merged) == 2
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 60.0
    assert merged[1]["start"] == 60.0


def test_dominant_tempo_in_interval_weighted():
    tempo_segments = [
        {"start": 0.0, "end": 50.0, "bpm": 120.0, "confidence": 0.9},
        {"start": 50.0, "end": 100.0, "bpm": 124.0, "confidence": 0.9},
    ]
    t = _dominant_tempo_in_interval(tempo_segments, 0.0, 100.0)
    assert 121.9 <= t["bpm"] <= 122.1
