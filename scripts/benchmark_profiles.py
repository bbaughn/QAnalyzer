#!/usr/bin/env python3
"""Compare chroma profile sets within our full pipeline.

Swaps only the major/minor profile weights in _key_from_chroma — all other
pipeline logic (P5 correction, global_root_idx heuristics, per-window
constrained analysis, dorian/mixolydian post-processing) stays identical.

Profiles tested:
  krumhansl  — current (Krumhansl 1990 + our mode extensions)
  bgate      — Beatport Generic (Faraldo 2017, zeros 4 least-relevant bins)
  braw       — Beatport Raw (Faraldo 2017, full corpus medians)
  edma       — Electronic Dance Music Auto (Faraldo 2016)

Usage:
    PYTHONPATH=. python3.12 scripts/benchmark_profiles.py
"""
import sys
from pathlib import Path

import yaml
import numpy as np

ROOT = Path(__file__).parent.parent
EXPECTATIONS = ROOT / "test_tracks" / "expectations.yaml"
CACHE_DIR = ROOT / "test_tracks" / "features_cache"
sys.path.insert(0, str(ROOT))

import app.services.analysis as analysis

# ---------------------------------------------------------------------------
# Profile definitions (from Essentia source, all starting on C, un-normalised)
# ---------------------------------------------------------------------------

# Our current Krumhansl major/minor weights (keep for reference)
_KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_BGATE_MAJOR = np.array([1.00, 0.00, 0.42, 0.00, 0.53, 0.37, 0.00, 0.77, 0.00, 0.38, 0.21, 0.30])
_BGATE_MINOR = np.array([1.00, 0.00, 0.36, 0.39, 0.00, 0.38, 0.00, 0.74, 0.27, 0.00, 0.42, 0.23])

_BRAW_MAJOR  = np.array([1.00, 0.1573, 0.420, 0.1570, 0.5296, 0.3669, 0.1632, 0.7711, 0.1676, 0.3827, 0.2113, 0.2965])
_BRAW_MINOR  = np.array([1.00, 0.2330, 0.3615, 0.3905, 0.2925, 0.3777, 0.1961, 0.7425, 0.2701, 0.2161, 0.4228, 0.2272])

_EDMA_MAJOR  = np.array([1.00, 0.29, 0.50, 0.40, 0.60, 0.56, 0.32, 0.80, 0.31, 0.45, 0.42, 0.39])
_EDMA_MINOR  = np.array([1.00, 0.31, 0.44, 0.58, 0.33, 0.49, 0.29, 0.78, 0.43, 0.29, 0.53, 0.32])

# libkeyfinder / Shaath (Ibrahim Shaath's thesis profiles — used by Mixxx)
_SHAATH_MAJOR = np.array([6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4])
_SHAATH_MINOR = np.array([6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2])

# Temperley 2005 (MIREX) — used as a strong baseline in many papers
_TEMP05_MAJOR = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
_TEMP05_MINOR = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])

# Gómez (as specified by Shaath) — sparse, bass-root-focused
_GOMEZ_MAJOR  = np.array([0.82, 0.00, 0.55, 0.00, 0.53, 0.30, 0.08, 1.00, 0.00, 0.38, 0.00, 0.47])
_GOMEZ_MINOR  = np.array([0.81, 0.00, 0.53, 0.54, 0.00, 0.27, 0.07, 1.00, 0.27, 0.07, 0.10, 0.36])

PROFILES = {
    "krumhansl": (_KRUMHANSL_MAJOR, _KRUMHANSL_MINOR),
    "shaath":    (_SHAATH_MAJOR,    _SHAATH_MINOR),
    "temp2005":  (_TEMP05_MAJOR,    _TEMP05_MINOR),
    "gomez":     (_GOMEZ_MAJOR,     _GOMEZ_MINOR),
    "bgate":     (_BGATE_MAJOR,     _BGATE_MINOR),
    "braw":      (_BRAW_MAJOR,      _BRAW_MINOR),
    "edma":      (_EDMA_MAJOR,      _EDMA_MINOR),
}

# Modes we keep unchanged when swapping major/minor
_EXTRA_MODES = [(name, prof) for name, prof in analysis.MODE_PROFILES
                if name not in ("major", "minor")]


def apply_profile(major: np.ndarray, minor: np.ndarray) -> None:
    """Patch analysis module's profile tables in-place."""
    new_profiles = [("major", major), ("minor", minor)] + _EXTRA_MODES
    analysis.MODE_PROFILES[:] = new_profiles
    analysis._MODE_PROFILES_NORM[:] = [
        (name, p / (np.linalg.norm(p) + 1e-9)) for name, p in new_profiles
    ]


# ---------------------------------------------------------------------------
# Comparison helpers (mirrors run_track_tests.py)
# ---------------------------------------------------------------------------

def compare(result: dict, expected: dict) -> list[tuple]:
    results = []
    g = result.get("global", {})
    sections = result.get("sections", [])

    def chk(label, actual, exp):
        results.append((actual == exp, label, exp, actual))

    chk("bars_percussion", g.get("bars_percussion_rounded"), expected.get("bars_percussion"))
    chk("no_drums", g.get("no_drums"), expected.get("no_drums"))
    if "no_key" in expected:
        chk("no_key", g.get("no_key", False), expected["no_key"])
    if "no_tempo" in expected:
        chk("no_tempo", g.get("no_tempo", False), expected["no_tempo"])
    chk("swing", g.get("swing"), expected.get("swing"))

    if sections and expected.get("key"):
        chk("key.root",  sections[0].get("key"),  expected["key"]["root"])
        chk("key.mode",  sections[0].get("mode"),  expected["key"]["mode"])
    if sections and expected.get("key_end"):
        chk("key_end.root", sections[-1].get("key"),  expected["key_end"]["root"])
        chk("key_end.mode", sections[-1].get("mode"),  expected["key_end"]["mode"])
    if sections and expected.get("bpm") is not None:
        chk("bpm",     sections[0].get("tempo_bpm_rounded"), int(expected["bpm"]))
    if sections and expected.get("bpm_end") is not None:
        chk("bpm_end", sections[-1].get("tempo_bpm_rounded"), int(expected["bpm_end"]))
    if sections and expected.get("tuning") is not None:
        chk("tuning",     sections[0].get("tuning_rounded"), int(expected["tuning"]))
    if sections and expected.get("tuning_end") is not None:
        chk("tuning_end", sections[-1].get("tuning_rounded"), int(expected["tuning_end"]))

    return results


def run_profile(profile_name: str) -> dict[str, bool]:
    """Run all tracks with the given profile. Returns {title: passed}."""
    major, minor = PROFILES[profile_name]
    apply_profile(major, minor)

    tracks = yaml.safe_load(EXPECTATIONS.read_text())["tracks"]
    results = {}
    for track in tracks:
        title = track["title"]
        wav = ROOT / track["file"]
        cache = CACHE_DIR / (wav.stem + ".json")
        if not cache.exists():
            continue
        features = analysis.load_features(str(cache))
        result = analysis.interpret_features(features)
        checks = compare(result, track["expected"])
        passed = all(ok for ok, *_ in checks)
        results[title] = passed
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    tracks = yaml.safe_load(EXPECTATIONS.read_text())["tracks"]
    all_titles = [t["title"] for t in tracks
                  if (CACHE_DIR / (Path(t["file"]).stem + ".json")).exists()]

    profile_results: dict[str, dict[str, bool]] = {}
    for name in PROFILES:
        print(f"  Running {name}…", flush=True)
        profile_results[name] = run_profile(name)

    # Summary table
    print(f"\n  {'Track':<44}", end="")
    for name in PROFILES:
        print(f"  {name:<10}", end="")
    print()
    print("  " + "─" * (44 + 12 * len(PROFILES)))

    totals = {name: 0 for name in PROFILES}
    profile_names = list(PROFILES.keys())

    for title in all_titles:
        row = f"  {title:<44}"
        for name in profile_names:
            passed = profile_results[name].get(title, None)
            if passed is None:
                mark = "  -"
            elif passed:
                mark = "  ✓"
                totals[name] += 1
            else:
                mark = "  ✗"
            row += f"  {mark:<10}"
        # Highlight rows where profiles disagree
        vals = [profile_results[name].get(title) for name in profile_names]
        if len(set(v for v in vals if v is not None)) > 1:
            row += "  ← differs"
        print(row)

    print("  " + "─" * (44 + 12 * len(PROFILES)))
    row = f"  {'TOTAL':<44}"
    for name in profile_names:
        row += f"  {totals[name]:<12}"
    print(row)

    # Detail: tracks where a new profile beats krumhansl or vice versa
    baseline = profile_results["krumhansl"]
    print()
    for name in profile_names[1:]:
        gains = [t for t in all_titles if not baseline.get(t) and profile_results[name].get(t)]
        losses = [t for t in all_titles if baseline.get(t) and not profile_results[name].get(t)]
        net = len(gains) - len(losses)
        print(f"  {name} vs krumhansl: {net:+d}  gains={gains}  losses={losses}")


if __name__ == "__main__":
    run()
