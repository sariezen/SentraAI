"""
Synthetic labeled-pair generator for the Sentra AI POC.

No real image pairs were delivered with the assignment, so this script builds a
small *labeled* dataset that lets us (a) demonstrate the detector end-to-end and
(b) measure accuracy / tune thresholds objectively.

It creates a synthetic "scene" (a static security-camera-like background) and
then produces two kinds of second frames:

  * LIGHTING pairs -- only illumination changes: global gain/bias, gamma, a soft
    feathered shadow, a low-frequency "cloud" brightness modulation, or a bright
    reflection spot. The physical scene is identical.

  * MOTION pairs   -- a small textured object is pasted into the scene (a person-
    like blob / box), optionally WITH its own cast shadow, to stress-test the
    detector against the hardest case (motion + lighting together).

Output: pairs written to images/ as  <label>_<id>_a.png / _b.png, plus a
labels.csv mapping each pair to its ground-truth label.

Usage:
    python synth.py                 # default: 6 lighting + 6 motion pairs
    python synth.py --n 10          # 10 of each
    python synth.py --out images    # output folder
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

# Deterministic output so results are reproducible across runs.
RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- #
# Scene construction                                                          #
# --------------------------------------------------------------------------- #

def build_scene(h: int = 360, w: int = 480) -> np.ndarray:
    """A static, textured grayscale-ish scene resembling a fixed camera view."""
    img = np.full((h, w, 3), 120, np.uint8)

    # Ground / wall split with slightly different base tones.
    img[: h // 2] = (95, 100, 110)        # "wall"
    img[h // 2 :] = (70, 78, 85)          # "ground"

    # Texture so the scene has real edges (gravel/bricks-like speckle + lines).
    speckle = RNG.integers(-18, 18, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + speckle, 0, 255).astype(np.uint8)

    # A few structural lines (a pole, a curb, a doorway) -> stable edges.
    cv2.line(img, (60, 0), (60, h), (160, 165, 170), 3)
    cv2.rectangle(img, (300, 40), (430, 175), (140, 140, 145), 2)
    cv2.line(img, (0, h // 2), (w, h // 2), (55, 60, 66), 2)
    for x in range(0, w, 40):
        cv2.line(img, (x, h // 2), (x + 12, h), (60, 66, 72), 1)

    return cv2.GaussianBlur(img, (3, 3), 0)


# --------------------------------------------------------------------------- #
# Lighting-only transformations (label = "Lighting Change")                   #
# --------------------------------------------------------------------------- #

def _global_gain_bias(img):
    gain = float(RNG.uniform(0.65, 1.45))
    bias = float(RNG.uniform(-35, 35))
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def _gamma(img):
    g = float(RNG.uniform(0.5, 1.8))
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _soft_shadow(img):
    """A large, soft, feathered darkening -- the classic false-alarm trigger."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cx, cy = RNG.integers(w // 4, 3 * w // 4), RNG.integers(h // 4, 3 * h // 4)
    ax, ay = RNG.integers(w // 4, w // 2), RNG.integers(h // 4, h // 2)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=45)  # heavy feather -> soft edge
    factor = 1.0 - 0.45 * mask[..., None]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _cloud(img):
    """Low-frequency multiplicative brightness field, like a passing cloud."""
    h, w = img.shape[:2]
    low = RNG.uniform(0.6, 1.3, (6, 8)).astype(np.float32)
    field = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=40)[..., None]
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)


def _reflection(img):
    """A bright, soft specular blob (sun glint / headlight reflection)."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    cx, cy = RNG.integers(0, w), RNG.integers(0, h)
    r = RNG.integers(w // 8, w // 4)
    blob = np.zeros((h, w), np.float32)
    cv2.circle(blob, (int(cx), int(cy)), int(r), 1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=r / 2)
    out += (blob[..., None] * RNG.uniform(60, 130))
    return np.clip(out, 0, 255).astype(np.uint8)


LIGHTING_OPS = [_global_gain_bias, _gamma, _soft_shadow, _cloud, _reflection]


# --------------------------------------------------------------------------- #
# Motion transformations (label = "Real Motion")                              #
# --------------------------------------------------------------------------- #

def _paste_object(img, with_shadow: bool):
    """Paste a small textured 'object' (person/box-like) into the scene."""
    out = img.copy()
    h, w = img.shape[:2]
    ow, oh = int(RNG.integers(28, 60)), int(RNG.integers(55, 110))
    x = int(RNG.integers(10, w - ow - 10))
    y = int(RNG.integers(h // 3, h - oh - 5))

    # Optional cast shadow first (so the object sits on top of it) -- this makes
    # the case realistic: real motion almost always brings some lighting change.
    if with_shadow:
        sh = out.astype(np.float32)
        smask = np.zeros((h, w), np.float32)
        cv2.ellipse(smask, (x + ow // 2 + 18, y + oh), (ow, oh // 3), 0, 0, 360, 1.0, -1)
        smask = cv2.GaussianBlur(smask, (0, 0), sigmaX=12)
        out = np.clip(sh * (1.0 - 0.4 * smask[..., None]), 0, 255).astype(np.uint8)

    # Textured object body: base color + internal speckle + outline + a couple of
    # internal lines -> plenty of NEW edges, which is the motion signature.
    base = RNG.integers(30, 220, 3).tolist()
    obj = np.full((oh, ow, 3), base, np.uint8)
    obj = np.clip(
        obj.astype(np.int16) + RNG.integers(-30, 30, (oh, ow, 3), np.int16), 0, 255
    ).astype(np.uint8)
    out[y : y + oh, x : x + ow] = obj
    cv2.rectangle(out, (x, y), (x + ow, y + oh), (20, 20, 20), 2)
    cv2.line(out, (x, y + oh // 2), (x + ow, y + oh // 2), (15, 15, 15), 1)
    cv2.line(out, (x + ow // 2, y), (x + ow // 2, y + oh), (15, 15, 15), 1)
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def generate(out_dir: str, n_each: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for i in range(n_each):
        scene = build_scene()
        op = LIGHTING_OPS[i % len(LIGHTING_OPS)]
        frame_b = op(scene)
        pid = f"lighting_{i:02d}"
        cv2.imwrite(os.path.join(out_dir, f"{pid}_a.png"), scene)
        cv2.imwrite(os.path.join(out_dir, f"{pid}_b.png"), frame_b)
        rows.append((pid, "Lighting Change", op.__name__))

    for i in range(n_each):
        scene = build_scene()
        with_shadow = bool(i % 2)  # half the motion pairs also include a shadow
        frame_b = _paste_object(scene, with_shadow)
        pid = f"motion_{i:02d}"
        cv2.imwrite(os.path.join(out_dir, f"{pid}_a.png"), scene)
        cv2.imwrite(os.path.join(out_dir, f"{pid}_b.png"), frame_b)
        rows.append((pid, "Real Motion", "object+shadow" if with_shadow else "object"))

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "ground_truth", "variant"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} pairs into '{out_dir}/' (labels -> {labels_path})")
    return labels_path


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic labeled image pairs.")
    ap.add_argument("--n", type=int, default=6, help="pairs per class (default 6)")
    ap.add_argument("--out", default="images", help="output folder (default images)")
    args = ap.parse_args()
    generate(args.out, args.n)


if __name__ == "__main__":
    main()
