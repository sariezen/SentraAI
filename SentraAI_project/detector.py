"""
Sentra AI POC -- Real Motion vs. Lighting Change detector.

Given two frames from the SAME static security camera, taken a few seconds
apart, decide whether the difference between them is caused by:

    * "Real Motion"     -- a physical object appeared / moved in the scene, or
    * "Lighting Change" -- only the illumination changed (shadow, cloud,
                           sunlight, reflection) with no physical change.

Core idea
---------
Lighting and motion differ in their *physical nature*, and we measure exactly
that difference instead of "how different the two images are":

    Lighting change  ->  photometric: changes pixel INTENSITY but PRESERVES
                         structure/edges; spatially smooth & diffuse; can be
                         modeled as a per-pixel gain + bias on intensity.

    Real motion      ->  geometric: a new object OCCLUDES the background, so it
                         ADDS / REMOVES edges and texture in a LOCALIZED,
                         COMPACT, textured region.

Pipeline
--------
    1. Preprocess (grayscale, denoise, optional alignment for camera jitter).
    2. Photometric normalization: fit I2 ~= a*I1 + b and compensate, so a pure
       lighting change largely vanishes and only structural change remains.
    3. Extract four complementary, interpretable cues (each ~0..1):
         C1 structural change  (SSIM map)
         C2 edge change        (illumination-invariant)
         C3 compensated residual (what is left after removing lighting)
         C4 spatial structure of the change region (compact+textured object vs
            diffuse+smooth shadow/cloud)
    4. Aggregate into a transparent "motion score", decide, and turn the signed
       distance from the decision boundary into a calibrated confidence 0..100%.

The result dict exposes every cue and a human-readable explanation, so the
decision is fully auditable -- important for a security product.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.metrics import structural_similarity


# --------------------------------------------------------------------------- #
# Tunable parameters (documented; kept in one place so they are easy to audit) #
# --------------------------------------------------------------------------- #

@dataclass
class DetectorConfig:
    # Preprocessing
    denoise_ksize: int = 3            # light Gaussian blur to suppress sensor noise
    align: bool = True               # try to absorb tiny sub-pixel camera jitter
    align_max_shift_frac: float = 0.05  # reject alignment that moves too much (not jitter)

    # Photometric normalization
    photo_robust_iters: int = 2       # robust (outlier-rejecting) gain+bias fit iterations

    # Cue thresholds
    residual_thresh: int = 25         # 0..255; what counts as a "real" leftover difference
    ssim_low: float = 0.55            # local SSIM below this = structurally changed pixel
    min_blob_area_frac: float = 0.0008  # ignore change blobs smaller than this (noise)

    # Decision weights. Evidence FOR motion is dominated by STRUCTURAL change
    # (new/lost edges). The "localized intensity change" cues (residual, SSIM
    # structure) only count as motion when accompanied by structural edges, so
    # they are GATED by texture rather than added blindly. w_global and w_smooth
    # SUBTRACT -- a uniform brightness shift, or a large but smooth changed blob,
    # are the signatures of a lighting change.
    w_texture: float = 2.6            # C4 structural (new/lost) edge density -- decisive
    w_edge: float = 1.6               # C2 global edge change
    w_localized: float = 1.7          # C3+C1 residual & SSIM structure, GATED by texture
    w_global: float = 2.0             # global brightness shift -> evidence FOR lighting
    w_smooth: float = 2.2             # smooth changed blob -> evidence FOR lighting
    gate_floor: float = 0.15          # min fraction of the localized cue that is ungated
    bias: float = -0.55               # decision boundary offset (score > 0 => motion)

    confidence_gain: float = 2.6      # sigmoid steepness for confidence calibration


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class ChangeResult:
    result: str                       # "Real Motion" | "Lighting Change"
    confidence: float                 # 0..100
    cues: dict = field(default_factory=dict)
    explanation: str = ""
    # Heavy artifacts for visualization (not printed); may be None.
    change_map: np.ndarray | None = None
    motion_mask: np.ndarray | None = None
    motion_score: float = 0.0

    def to_public_dict(self) -> dict:
        """Lightweight dict (no images) -- safe to print / write to CSV."""
        return {
            "result": self.result,
            "confidence": round(self.confidence, 1),
            "motion_score": round(self.motion_score, 3),
            **{k: round(float(v), 3) for k, v in self.cues.items()},
        }


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _ensure_same_size(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape[:2] != b.shape[:2]:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h))
        b = cv2.resize(b, (w, h))
    return a, b


# --------------------------------------------------------------------------- #
# Pipeline stages                                                             #
# --------------------------------------------------------------------------- #

def _preprocess(img1: np.ndarray, img2: np.ndarray, cfg: DetectorConfig):
    """Grayscale + denoise + optional alignment. Returns float32 images in 0..255."""
    g1, g2 = _to_gray(img1), _to_gray(img2)
    g1, g2 = _ensure_same_size(g1, g2)

    k = cfg.denoise_ksize
    if k and k >= 3:
        g1 = cv2.GaussianBlur(g1, (k, k), 0)
        g2 = cv2.GaussianBlur(g2, (k, k), 0)

    aligned = False
    if cfg.align:
        try:
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            # Align img1 onto img2 with a translation-only model (pure jitter).
            _, warp = cv2.findTransformECC(
                g2.astype(np.float32), g1.astype(np.float32),
                warp, cv2.MOTION_TRANSLATION, criteria, None, 5,
            )
            max_shift = cfg.align_max_shift_frac * max(g1.shape)
            if abs(warp[0, 2]) <= max_shift and abs(warp[1, 2]) <= max_shift:
                g1 = cv2.warpAffine(
                    g1, warp, (g1.shape[1], g1.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                aligned = True
        except cv2.error:
            # Alignment is a best-effort refinement; failure is non-fatal.
            aligned = False

    return g1.astype(np.float32), g2.astype(np.float32), aligned


def _photometric_normalize(g1: np.ndarray, g2: np.ndarray, cfg: DetectorConfig):
    """
    Fit a global gain+bias model  g2 ~= a*g1 + b  with robust (outlier-rejecting)
    least squares, then map g1 into g2's illumination. After this, a pure lighting
    change between the frames is largely removed -- whatever residual remains is
    structural (i.e. evidence of real motion).

    Returns (g1_compensated, gain a, bias b, global_shift in 0..1).
    """
    x = g1.ravel()
    y = g2.ravel()
    weights = np.ones_like(x)

    a, b = 1.0, 0.0
    for _ in range(max(1, cfg.photo_robust_iters)):
        w = weights
        sw = w.sum()
        mx = (w * x).sum() / sw
        my = (w * y).sum() / sw
        cov = (w * (x - mx) * (y - my)).sum()
        var = (w * (x - mx) ** 2).sum()
        a = cov / var if var > 1e-6 else 1.0
        b = my - a * mx
        # Re-weight: pixels that disagree with the global lighting model (likely
        # the moving object) get down-weighted so they do not bias the fit.
        resid = np.abs(y - (a * x + b))
        scale = np.median(resid) + 1e-6
        weights = 1.0 / (1.0 + (resid / (2.5 * scale)) ** 2)

    g1_comp = a * g1 + b

    # A strong global brightness move (a far from 1 and/or large |b|) is itself a
    # signature of a lighting change. Summarize it as a single 0..1 number.
    global_shift = np.clip(abs(np.log(max(a, 1e-3))) * 0.7 + abs(b) / 255.0, 0.0, 1.0)
    return g1_comp, float(a), float(b), float(global_shift)


def _cue_structure(g1: np.ndarray, g2: np.ndarray, cfg: DetectorConfig):
    """C1 -- localized structural change via the SSIM map (illumination-aware)."""
    score, ssim_map = structural_similarity(
        g1, g2, data_range=255.0, full=True, gaussian_weights=True, sigma=1.5,
    )
    dissim = 1.0 - ssim_map                       # 0 = identical, 1 = totally different
    changed = (ssim_map < cfg.ssim_low)
    frac_changed = float(changed.mean())
    # Localized structural change: how much dissimilarity sits in the changed
    # pixels, scaled so a small-but-strong region still registers.
    localized = float(dissim[changed].mean()) if changed.any() else 0.0
    localized = localized * np.sqrt(min(frac_changed * 50.0, 1.0))
    return {
        "ssim": float(score),
        "ssim_frac_changed": frac_changed,
        "structure": float(np.clip(localized, 0.0, 1.0)),
    }, dissim


def _edge_map(g: np.ndarray) -> np.ndarray:
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    med = float(np.median(g8))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    return cv2.Canny(g8, lo, hi) > 0


def _cue_edges(g1: np.ndarray, g2: np.ndarray, cfg: DetectorConfig):
    """
    C2 -- edge change. Edges are largely illumination-invariant, so NEW or
    VANISHED edges indicate a real structural change rather than lighting.
    Measured as 1 - IoU between the two edge maps.
    """
    e1, e2 = _edge_map(g1), _edge_map(g2)
    inter = np.logical_and(e1, e2).sum()
    union = np.logical_or(e1, e2).sum()
    edge_iou = float(inter / union) if union > 0 else 1.0
    edge_change = float(np.clip(1.0 - edge_iou, 0.0, 1.0))
    return {"edge_iou": edge_iou, "edge_change": edge_change}, (e1, e2)


def _cue_residual_and_structure(
    g1_comp: np.ndarray, g2: np.ndarray, edges: tuple[np.ndarray, np.ndarray],
    cfg: DetectorConfig,
):
    """
    C3 + C4 -- after lighting is removed, analyze the leftover difference:

      C3 residual: how strong & compact is the leftover signal.
      C4 texture : edge density INSIDE the change region. A real object is
                   textured (many edges); a shadow/cloud patch is smooth.

    Also returns a normalized change map and the binary motion mask for
    visualization.
    """
    diff = np.abs(g2 - g1_comp)
    change_map = np.clip(diff / 255.0, 0.0, 1.0)

    raw_mask = (diff > cfg.residual_thresh).astype(np.uint8)
    # Clean up: close small gaps, drop tiny noise blobs.
    raw_mask = cv2.morphologyEx(
        raw_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    raw_mask = cv2.morphologyEx(
        raw_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )

    total_px = raw_mask.size
    min_area = cfg.min_blob_area_frac * total_px

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, 8)
    motion_mask = np.zeros_like(raw_mask)
    kept_area = 0
    largest_area = 0
    largest_solidity = 0.0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        motion_mask[labels == i] = 1
        kept_area += area
        if area > largest_area:
            largest_area = area
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bbox = max(w * h, 1)
            largest_solidity = area / bbox  # fill ratio of its bounding box

    changed_frac = kept_area / total_px

    # C3: residual strength concentrated in a compact, well-filled region.
    if kept_area > 0:
        resid_strength = float((diff[motion_mask > 0]).mean() / 255.0)
    else:
        resid_strength = 0.0
    # Reward presence of a solid, non-trivial region; penalize "spread thin".
    residual_cue = resid_strength * float(largest_solidity)
    residual_cue *= np.sqrt(min(changed_frac * 80.0, 1.0))
    residual_cue = float(np.clip(residual_cue, 0.0, 1.0))

    # C4: STRUCTURAL edge change inside the change region. The decisive cue.
    # A real object ADDS edges (its outline/texture) and REMOVES background edges
    # it occludes. A local shadow or reflection is SMOOTH -- it leaves the edges
    # underneath essentially intact, so it produces almost no new/lost edges.
    # We therefore measure NEW (in B, not A) + LOST (in A, not B) edges -- not the
    # raw union, which would also count the static background texture sitting
    # under a shadow and falsely look "textured".
    e1, e2 = edges
    de1 = cv2.dilate(e1.astype(np.uint8), np.ones((3, 3), np.uint8))  # tolerance
    de2 = cv2.dilate(e2.astype(np.uint8), np.ones((3, 3), np.uint8))
    new_edges = np.logical_and(e2, de1 == 0)
    lost_edges = np.logical_and(e1, de2 == 0)
    struct_edges = np.logical_or(new_edges, lost_edges)

    if kept_area > 0:
        struct_density = float(struct_edges[motion_mask > 0].mean())
    else:
        struct_density = 0.0
    # ~6% structural-edge pixels inside the region already means a clearly
    # textured object; scale that to ~1.0.
    texture_cue = float(np.clip(struct_density / 0.06, 0.0, 1.0))

    # "Smooth blob": a non-trivial changed region that carries almost no
    # structural edges -- the signature of a local lighting change (shadow /
    # reflection / cloud edge). Used to actively pull the decision toward
    # "Lighting Change".
    smooth_blob = float(np.sqrt(min(changed_frac * 80.0, 1.0)) * (1.0 - texture_cue))

    cues = {
        "changed_area_frac": float(changed_frac),
        "largest_solidity": float(largest_solidity),
        "residual": residual_cue,
        "texture": texture_cue,
        "smooth_blob": smooth_blob,
    }
    return cues, change_map, motion_mask


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def classify_change(
    img1: np.ndarray, img2: np.ndarray, cfg: DetectorConfig | None = None,
) -> ChangeResult:
    """
    Classify the change between two BGR (or grayscale) images.

    Returns a ChangeResult with the label, a 0..100 confidence, every cue value,
    a human-readable explanation, and visualization artifacts.
    """
    cfg = cfg or DetectorConfig()

    g1, g2, aligned = _preprocess(img1, img2, cfg)
    g1_comp, gain, bias, global_shift = _photometric_normalize(g1, g2, cfg)

    c1, _dissim = _cue_structure(g1, g2, cfg)
    c2, edges = _cue_edges(g1, g2, cfg)
    c34, change_map, motion_mask = _cue_residual_and_structure(g1_comp, g2, edges, cfg)

    cues = {**c1, **c2, **c34, "global_shift": global_shift}

    # "Localized intensity change" cues (compensated residual + SSIM structure
    # drop) signal that *something* changed in a spot -- but a local shadow does
    # that too. So we GATE them by texture: they only count as motion evidence in
    # proportion to how much structural edge change accompanies them.
    localized = 0.5 * (cues["residual"] + cues["structure"])
    gate = cfg.gate_floor + (1.0 - cfg.gate_floor) * cues["texture"]
    localized_gated = localized * gate

    # Transparent weighted "motion score". Positive => motion, negative => lighting.
    motion_score = (
        cfg.w_texture * cues["texture"]
        + cfg.w_edge * cues["edge_change"]
        + cfg.w_localized * localized_gated
        - cfg.w_global * cues["global_shift"]
        - cfg.w_smooth * cues["smooth_blob"]
        + cfg.bias
    )

    is_motion = motion_score > 0.0
    confidence = 100.0 * _sigmoid(cfg.confidence_gain * abs(motion_score))
    # Keep confidence honest near the boundary (never claim 100% certainty).
    confidence = float(np.clip(confidence, 50.0, 99.0))

    result = "Real Motion" if is_motion else "Lighting Change"
    explanation = _explain(result, cues, gain, bias, aligned)

    return ChangeResult(
        result=result,
        confidence=confidence,
        cues=cues,
        explanation=explanation,
        change_map=change_map,
        motion_mask=motion_mask,
        motion_score=float(motion_score),
    )


def _explain(result: str, cues: dict, gain: float, bias: float, aligned: bool) -> str:
    """Build a short, human-readable justification from the dominant cues."""
    parts: list[str] = []
    if result == "Real Motion":
        if cues["edge_change"] > 0.15:
            parts.append(f"new/removed edges (edge change {cues['edge_change']:.2f})")
        if cues["texture"] > 0.2:
            parts.append(f"textured change region (texture {cues['texture']:.2f})")
        if cues["residual"] > 0.1:
            parts.append(
                f"compact leftover after lighting compensation "
                f"(residual {cues['residual']:.2f}, "
                f"solidity {cues['largest_solidity']:.2f})"
            )
        if cues["structure"] > 0.1:
            parts.append(f"localized structural change (structure {cues['structure']:.2f})")
        reason = "; ".join(parts) if parts else "weak but net structural evidence"
        head = "Physical change detected: "
    else:
        parts.append(
            f"global brightness shift gain={gain:.2f}, bias={bias:.1f} "
            f"(global_shift {cues['global_shift']:.2f})"
        )
        if cues["edge_change"] < 0.15:
            parts.append(f"edges preserved (edge change {cues['edge_change']:.2f})")
        if cues["residual"] < 0.12:
            parts.append("little compact residual after lighting compensation")
        reason = "; ".join(parts)
        head = "No physical change -- illumination only: "

    note = " [frames aligned for jitter]" if aligned else ""
    return head + reason + note
