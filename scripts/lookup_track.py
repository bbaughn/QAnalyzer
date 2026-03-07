#!/usr/bin/env python3
"""Look up a track in tracks.csv and return analyzer-comparable data.

Usage:
    python scripts/lookup_track.py "Crying Juno"
    python scripts/lookup_track.py "Rain and Snow Mixture"

Key notation used in the CSV:
    Suffix  Mode
    (none)  major
    m       minor
    d       dorian
    x       mixolydian
    p       phrygian
    l       minor  (locrian treated as minor, matching analyzer behavior)

Examples: Cm → C minor, Bbd → Bb dorian, Ebm → Eb minor, G → G major
"""

import csv
import json
import re
import sys
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "tracks.csv"

_MODE_SUFFIX = {
    "m": "minor",
    "d": "dorian",
    "x": "mixolydian",
    "p": "phrygian",
    "l": "minor",  # locrian → minor
}


def parse_key(s: str) -> dict | None:
    s = (s or "").strip()
    if not s or s.lower() == "none":
        return None
    suffix = s[-1]
    if suffix in _MODE_SUFFIX:
        return {"root": s[:-1], "mode": _MODE_SUFFIX[suffix]}
    return {"root": s, "mode": "major"}


def parse_perc(s: str) -> int:
    """Strip annotations like '!' and return integer bar count."""
    digits = re.sub(r"[^0-9.]", "", (s or "").strip())
    return int(float(digits)) if digits else 0


def parse_bool(s: str) -> bool:
    return (s or "").strip().upper() == "TRUE"


def parse_float(s: str) -> float | None:
    try:
        return float((s or "").strip())
    except ValueError:
        return None


def lookup_row(title: str) -> dict | None:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Title", "").strip().lower() == title.strip().lower():
                return row
    return None


def format_track(row: dict) -> dict:
    """Return only the fields the analyzer can be compared against."""
    return {
        "artist": row.get("Artist", "").strip(),
        "title": row.get("Title", "").strip(),
        "bpm": parse_float(row.get("BPM", "")),
        "bpm_end": parse_float(row.get("BPM End", "")),
        "key": parse_key(row.get("Key", "")),
        "key_end": parse_key(row.get("Key End", "")),
        "tuning": int(parse_float(row.get("Tune", "0")) or 0),
        "tuning_end": int(parse_float(row.get("Tune End", "0")) or 0),
        "no_drums": parse_bool(row.get("No Drums", "")),
        "bars_percussion": parse_perc(row.get("Perc", "0")),
        "swing": parse_bool(row.get("Swing", "")),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: lookup_track.py <title>", file=sys.stderr)
        sys.exit(1)
    title = " ".join(sys.argv[1:])
    row = lookup_row(title)
    if row is None:
        print(f"Track not found: {title!r}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(format_track(row), indent=2))
