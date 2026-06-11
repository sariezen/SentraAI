"""
Sentra AI POC -- command-line entry point for a single image pair.

Usage:
    python main.py <image1> <image2> [--save out.png] [--no-show]

Prints the verdict (Real Motion / Lighting Change), a confidence score (0..100%),
the per-cue breakdown, and a one-line human-readable explanation. Optionally saves
a visualization panel (originals + change heatmap + detected change region).
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from detector import classify_change, ChangeResult


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"ERROR: could not read image '{path}'")
    return img


def build_visualization(img1, img2, res: ChangeResult) -> np.ndarray:
    """A 2x2 panel: frame A, frame B, change heatmap, detected change overlay."""
    h, w = img1.shape[:2]
    img2r = cv2.resize(img2, (w, h)) if img2.shape[:2] != (h, w) else img2

    heat = (res.change_map * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    overlay = img2r.copy()
    mask = (res.motion_mask > 0).astype(np.uint8)
    if mask.any():
        color = (0, 0, 255) if res.result == "Real Motion" else (0, 200, 255)
        tint = np.zeros_like(overlay)
        tint[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.45, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    top = np.hstack([label(img1, "A (reference)"), label(img2r, "B (current)")])
    bot = np.hstack([label(heat, "change heatmap"),
                     label(overlay, "detected change region")])
    panel = np.vstack([top, bot])

    # Verdict banner at the bottom.
    banner = np.zeros((54, panel.shape[1], 3), np.uint8)
    color = (60, 60, 255) if res.result == "Real Motion" else (60, 200, 255)
    cv2.putText(banner, f"{res.result}   confidence {res.confidence:.0f}%",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(banner, res.explanation[:110], (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([panel, banner])


def print_report(res: ChangeResult) -> None:
    print("=" * 60)
    print(f"  Result     : {res.result}")
    print(f"  Confidence : {res.confidence:.0f}%")
    print("-" * 60)
    print("  Cues:")
    for k, v in res.cues.items():
        print(f"    {k:<20} {v:.3f}")
    print("-" * 60)
    print(f"  Why: {res.explanation}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Classify change between two camera frames.")
    ap.add_argument("image1")
    ap.add_argument("image2")
    ap.add_argument("--save", default=None, help="path to save the visualization PNG")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    args = ap.parse_args()

    img1, img2 = _read(args.image1), _read(args.image2)
    res = classify_change(img1, img2)

    print_report(res)

    panel = build_visualization(img1, img2, res)
    save_path = args.save
    if save_path is None:
        os.makedirs("outputs", exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.image1))[0]
        save_path = os.path.join("outputs", f"{stem}_result.png")
    cv2.imwrite(save_path, panel)
    print(f"Visualization saved to: {save_path}")

    if not args.no_show:
        try:
            cv2.imshow("Sentra AI -- change classification", panel)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # headless environment -> the saved PNG is the deliverable


if __name__ == "__main__":
    main()
