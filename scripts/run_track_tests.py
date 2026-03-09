#!/usr/bin/env python3
"""Run expectation tests against the analyzer's interpretation phase.

Audio features are extracted once and cached as JSON alongside each wav file.
Subsequent runs skip phase 1 entirely and only re-run interpret_features(),
so threshold changes can be validated in seconds.

Usage:
    # First run (or after changing the audio file): extracts and caches features
    PYTHONPATH=. .venv/bin/python3.12 scripts/run_track_tests.py

    # Fast re-run after threshold/config changes (uses cached features):
    PYTHONPATH=. .venv/bin/python3.12 scripts/run_track_tests.py

    # Force re-extraction even if cache exists:
    PYTHONPATH=. .venv/bin/python3.12 scripts/run_track_tests.py --reextract
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
EXPECTATIONS = ROOT / "test_tracks" / "expectations.yaml"
CACHE_DIR = ROOT / "test_tracks" / "features_cache"

sys.path.insert(0, str(ROOT))

from app.services.analysis import extract_audio_features, interpret_features, load_features, save_features


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _check(label: str, actual, expected, results: list) -> None:
    ok = actual == expected
    results.append((ok, label, expected, actual))


def compare(result: dict, expected: dict) -> list[tuple]:
    """Return list of (passed, label, expected, actual) tuples."""
    results = []
    g = result.get("global", {})
    sections = result.get("sections", [])

    # bars_percussion
    _check("bars_percussion", g.get("bars_percussion_rounded"), expected.get("bars_percussion"), results)

    # no_drums
    _check("no_drums", g.get("no_drums"), expected.get("no_drums"), results)

    # no_key
    if "no_key" in expected:
        _check("no_key", g.get("no_key", False), expected["no_key"], results)

    # no_tempo
    if "no_tempo" in expected:
        _check("no_tempo", g.get("no_tempo", False), expected["no_tempo"], results)

    # swing
    _check("swing", g.get("swing"), expected.get("swing"), results)

    # key / mode of first section
    if sections and expected.get("key"):
        first = sections[0]
        _check("key.root", first.get("key"), expected["key"]["root"], results)
        _check("key.mode", first.get("mode"), expected["key"]["mode"], results)

    # key / mode of last section
    if sections and expected.get("key_end"):
        last = sections[-1]
        _check("key_end.root", last.get("key"), expected["key_end"]["root"], results)
        _check("key_end.mode", last.get("mode"), expected["key_end"]["mode"], results)

    # bpm of first section (rounded)
    if sections and expected.get("bpm") is not None:
        _check("bpm", sections[0].get("tempo_bpm_rounded"), int(expected["bpm"]), results)

    # bpm of last section (rounded)
    if sections and expected.get("bpm_end") is not None:
        _check("bpm_end", sections[-1].get("tempo_bpm_rounded"), int(expected["bpm_end"]), results)

    # tuning of first section (rounded)
    if sections and expected.get("tuning") is not None:
        _check("tuning", sections[0].get("tuning_rounded"), int(expected["tuning"]), results)

    # tuning of last section (rounded)
    if sections and expected.get("tuning_end") is not None:
        _check("tuning_end", sections[-1].get("tuning_rounded"), int(expected["tuning_end"]), results)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(reextract: bool = False) -> bool:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tracks = yaml.safe_load(EXPECTATIONS.read_text())["tracks"]

    all_passed = True

    for track in tracks:
        title = track["title"]
        wav = ROOT / track["file"]
        cache = CACHE_DIR / (wav.stem + ".json")

        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"  {wav.name}")

        # Phase 1 — extract or load cached features
        if cache.exists() and not reextract:
            print(f"  [cache] loading features from {cache.name}")
            features = load_features(str(cache))
        else:
            print(f"  [phase 1] extracting audio features …")
            features = extract_audio_features(str(wav))
            save_features(features, str(cache))
            print(f"  [phase 1] done — cached to {cache.name}")

        # Phase 2 — interpret (fast)
        result = interpret_features(features)

        # Compare
        checks = compare(result, track["expected"])
        passed = sum(1 for ok, *_ in checks if ok)
        total = len(checks)
        track_ok = passed == total
        all_passed = all_passed and track_ok

        for ok, label, exp, act in checks:
            icon = "✓" if ok else "✗"
            if ok:
                print(f"    {icon} {label}: {act}")
            else:
                print(f"    {icon} {label}: expected={exp!r}  got={act!r}")

        print(f"  {'PASS' if track_ok else 'FAIL'} ({passed}/{total})")

    print(f"\n{'═' * 60}")
    print(f"  {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reextract", action="store_true", help="Re-run phase 1 even if cache exists")
    args = parser.parse_args()
    sys.exit(0 if run(reextract=args.reextract) else 1)
