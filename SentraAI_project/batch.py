"""
Sentra AI POC -- batch runner over a folder of image pairs.

Discovers pairs in a folder by the naming convention  <id>_a.<ext> / <id>_b.<ext>
(the convention produced by synth.py; real pairs can follow it too), classifies
each one, writes outputs/results.csv, saves a visualization per pair, and -- if a
labels.csv with ground truth is present -- prints an accuracy summary.

Usage:
    python batch.py [images_dir] [--out outputs] [--no-viz]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2

from detector import classify_change
from main import build_visualization


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_pairs(folder: str) -> list[tuple[str, str, str]]:
    """Return [(pair_id, path_a, path_b)] for files named <id>_a.* / <id>_b.*."""
    pairs = []
    for a_path in sorted(glob.glob(os.path.join(folder, "*_a.*"))):
        if not a_path.lower().endswith(IMG_EXTS):
            continue
        stem = a_path[: -len("_a" + os.path.splitext(a_path)[1])]
        pair_id = os.path.basename(stem)
        # Find the matching _b with any supported extension.
        b_candidates = [
            stem + "_b" + ext for ext in IMG_EXTS
            if os.path.exists(stem + "_b" + ext)
        ]
        if b_candidates:
            pairs.append((pair_id, a_path, b_candidates[0]))
    return pairs


def load_ground_truth(folder: str) -> dict[str, str]:
    path = os.path.join(folder, "labels.csv")
    gt = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                gt[row["pair_id"]] = row["ground_truth"]
    return gt


def main():
    ap = argparse.ArgumentParser(description="Batch-classify image pairs in a folder.")
    ap.add_argument("images_dir", nargs="?", default="images")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-viz", action="store_true", help="skip saving visualizations")
    args = ap.parse_args()

    pairs = find_pairs(args.images_dir)
    if not pairs:
        raise SystemExit(
            f"No pairs found in '{args.images_dir}'. Expected files like "
            f"'<id>_a.png' and '<id>_b.png'. Run  python synth.py  first."
        )

    os.makedirs(args.out, exist_ok=True)
    gt = load_ground_truth(args.images_dir)

    rows = []
    correct = 0
    evaluated = 0

    print(f"{'pair_id':<16}{'prediction':<18}{'conf':>6}  {'truth':<18}{'ok'}")
    print("-" * 64)

    for pair_id, pa, pb in pairs:
        img1, img2 = cv2.imread(pa, cv2.IMREAD_COLOR), cv2.imread(pb, cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            print(f"{pair_id:<16}(unreadable)")
            continue

        res = classify_change(img1, img2)
        truth = gt.get(pair_id, "")
        ok = ""
        if truth:
            evaluated += 1
            hit = (truth == res.result)
            correct += int(hit)
            ok = "OK" if hit else "XX"

        print(f"{pair_id:<16}{res.result:<18}{res.confidence:>5.0f}%  "
              f"{truth:<18}{ok}")

        row = {"pair_id": pair_id, **res.to_public_dict(), "ground_truth": truth,
               "explanation": res.explanation}
        rows.append(row)

        if not args.no_viz:
            panel = build_visualization(img1, img2, res)
            cv2.imwrite(os.path.join(args.out, f"{pair_id}_result.png"), panel)

    csv_path = os.path.join(args.out, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 64)
    print(f"Wrote {len(rows)} results to {csv_path}")
    if not args.no_viz:
        print(f"Saved visualizations to {args.out}/")
    if evaluated:
        acc = 100.0 * correct / evaluated
        print(f"Accuracy on labeled pairs: {correct}/{evaluated} = {acc:.1f}%")


if __name__ == "__main__":
    main()
