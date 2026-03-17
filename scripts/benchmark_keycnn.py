#!/usr/bin/env python3
"""Benchmark Key-CNN and evaluate a margin-gated ensemble.

Strategy: only let CNN influence global_root_idx when the chroma root signal
is *ambiguous* (top-1 minus top-2 root score < threshold).  When chroma is
clear, trust it; when chroma is uncertain, let CNN break the tie.

Usage:
    PYTHONPATH=. python3.12 scripts/benchmark_keycnn.py
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
EXPECTATIONS = ROOT / "test_tracks" / "expectations.yaml"
CACHE_DIR = ROOT / "test_tracks" / "features_cache"
sys.path.insert(0, str(ROOT))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import librosa
import tf_keras as keras

MODEL_PATH = "/tmp/key-cnn/keycnn/models/deepspec_k24.h5"

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {
    "C#": "Db", "Db": "C#", "D#": "Eb", "Eb": "D#",
    "F#": "Gb", "Gb": "F#", "G#": "Ab", "Ab": "G#",
    "A#": "Bb", "Bb": "A#",
}
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJOR_N = _MAJOR / np.linalg.norm(_MAJOR)
_MINOR_N = _MINOR / np.linalg.norm(_MINOR)

# Ground-truth pipeline pass/fail from run_track_tests
PIPELINE_PASSING = {
    "Crying Juno", "Subterano", "Loose Jaw", "Flute", "Slios", "Glycol",
    "Melchom", "Lindwurm", "Incompleteness", "It's Hard To Say", "Pygmy",
    "Rain and Snow Mixture", "Port Ana", "Alpaca", "Nocturnal",
    "To The Edge of the World", "Psychotic Particle", "Deflection",
}


def roots_match(a, b):
    return a == b or ENHARMONIC.get(a) == b


def chroma_root_scores_12(chroma_vec: np.ndarray) -> np.ndarray:
    cn = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-9)
    scores = np.zeros(12)
    for r in range(12):
        scores[r] = max(
            float(np.dot(cn, np.roll(_MAJOR_N, r))),
            float(np.dot(cn, np.roll(_MINOR_N, r))),
        )
    return scores


def std_normalizer(data):
    data = data.astype(np.float64)
    mean, std = np.mean(data), np.std(data)
    if std != 0.:
        data = (data - mean) / std
    return data.astype(np.float16)


def read_cqt_features(path: str, frames=60, hop_length=30) -> np.ndarray:
    y, sr = librosa.load(path, sr=22050)
    data = np.abs(librosa.cqt(
        y, sr=sr, hop_length=8192 // 2,
        fmin=librosa.note_to_hz("E1"),
        n_bins=168, bins_per_octave=24,
    ))
    data = np.reshape(data, (1, data.shape[0], data.shape[1], 1))
    if data.shape[2] < frames:
        padded = np.zeros((1, data.shape[1], frames, 1), dtype=data.dtype)
        padded[0, :, :data.shape[2], 0] = data[0, :, :, 0]
        data = padded
    windows = []
    total = data.shape[2]
    for offset in range(0, ((total - frames) // hop_length + 1) * hop_length, hop_length):
        windows.append(np.copy(data[:, :, offset:frames + offset, :]))
    return np.concatenate(windows, axis=0)


def cnn_probs_24(model, path: str) -> np.ndarray:
    import tensorflow as tf
    features = read_cqt_features(path)
    normed = std_normalizer(features).astype(np.float32)
    preds = model(tf.constant(normed), training=False).numpy()
    return np.mean(preds, axis=0)


def cnn_root_scores_12(probs24: np.ndarray) -> np.ndarray:
    return probs24[:12] + probs24[12:]


def margin_gated_root(c12: np.ndarray, n12: np.ndarray, threshold: float) -> str:
    """Use CNN only when chroma margin < threshold; otherwise trust chroma."""
    sorted_c = np.sort(c12)[::-1]
    margin = float(sorted_c[0] - sorted_c[1])
    if margin < threshold:
        # Ambiguous chroma — use multiplicative ensemble
        combined = c12 * n12
        return KEYS[int(np.argmax(combined))]
    else:
        return KEYS[int(np.argmax(c12))]


def run():
    print("Loading deepspec_k24…")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.\n")

    tracks = yaml.safe_load(EXPECTATIONS.read_text())["tracks"]

    # Collect per-track data
    rows = []
    for track in tracks:
        title = track["title"]
        wav = ROOT / track["file"]
        cache = CACHE_DIR / (wav.stem + ".json")
        expected = track["expected"]

        if not wav.exists() or not cache.exists():
            continue
        if expected.get("no_key") or expected.get("no_tempo"):
            continue
        exp_key = expected.get("key")
        if not exp_key:
            continue

        exp_root = exp_key["root"]
        passing = title in PIPELINE_PASSING

        try:
            features = json.loads(cache.read_text())
            probs24 = cnn_probs_24(model, str(wav))
        except Exception as e:
            print(f"  ERROR {title}: {e!r:.80}")
            continue

        global_chroma = np.mean(np.array(features["chroma_sync"]), axis=1)
        c12 = chroma_root_scores_12(global_chroma)
        n12 = cnn_root_scores_12(probs24)

        sorted_c = np.sort(c12)[::-1]
        margin = float(sorted_c[0] - sorted_c[1])

        chroma_root = KEYS[int(np.argmax(c12))]
        cnn_root = KEYS[int(np.argmax(n12))]
        cnn_conf = float(np.max(probs24))

        chroma_ok = roots_match(chroma_root, exp_root)
        cnn_ok = roots_match(cnn_root, exp_root)

        rows.append({
            "title": title, "exp_root": exp_root, "passing": passing,
            "chroma_root": chroma_root, "chroma_ok": chroma_ok, "margin": margin,
            "cnn_root": cnn_root, "cnn_ok": cnn_ok, "cnn_conf": cnn_conf,
            "c12": c12, "n12": n12,
        })

    # Per-track detail table
    print(f"  {'Track':<42} {'Exp':<4} {'Chroma':<8} {'Margin':<8} {'CNN':<8} {'CConf':<6} {'Pipeline'}")
    print("  " + "─" * 88)
    for r in sorted(rows, key=lambda x: x["margin"]):
        chroma_mark = "✓" if r["chroma_ok"] else "✗"
        cnn_mark = "✓" if r["cnn_ok"] else "✗"
        pipeline = "PASS" if r["passing"] else "FAIL"
        print(f"  {r['title']:<42} {r['exp_root']:<4} {chroma_mark}{r['chroma_root']:<7} {r['margin']:.4f}  {cnn_mark}{r['cnn_root']:<7} {r['cnn_conf']:.2f}   {pipeline}")

    # Sweep thresholds
    print(f"\n  Threshold sweep (margin < T → use CNN ensemble):")
    print(f"  {'T':>6}  {'Gains':>6} {'Regr':>6} {'Net':>6}  gains / regressions")
    print("  " + "─" * 80)

    best_net = -999
    best_t = None

    for t_int in range(0, 61, 2):
        t = t_int / 1000.0  # 0.000 to 0.060

        gains, regressions = [], []
        for r in rows:
            root = margin_gated_root(r["c12"], r["n12"], threshold=t)
            ok = roots_match(root, r["exp_root"])
            if ok and not r["passing"]:
                gains.append(r["title"])
            elif not ok and r["passing"]:
                regressions.append(r["title"])

        net = len(gains) - len(regressions)
        if net > best_net:
            best_net = net
            best_t = t

        marker = " ◄ best" if t == best_t or (net == best_net and best_t == t) else ""
        if net > 0 or t in (0.0, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060):
            gain_names = ", ".join(gains[:3]) + ("…" if len(gains) > 3 else "")
            regr_names = ", ".join(regressions[:3]) + ("…" if len(regressions) > 3 else "")
            print(f"  {t:.3f}  {len(gains):>6} {len(regressions):>6} {net:>+6}  +[{gain_names}]  -[{regr_names}]{marker}")

    # Best threshold detail
    print(f"\n  Best threshold T={best_t:.3f} (net {best_net:+d}):")
    gains, regressions = [], []
    for r in rows:
        root = margin_gated_root(r["c12"], r["n12"], threshold=best_t)
        ok = roots_match(root, r["exp_root"])
        if ok and not r["passing"]:
            gains.append(r["title"])
        elif not ok and r["passing"]:
            regressions.append(r["title"])
    print(f"  GAINS: {gains}")
    print(f"  REGRESSIONS: {regressions}")


if __name__ == "__main__":
    run()
