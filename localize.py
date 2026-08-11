"""
Drift-Sense: AI-Powered Localization Engine (v6.0 — Rotation & Scale Robust)
===========================================================================
Architecture:
  1. Multi-scale NCC sweep (8.3x-12.5x, covers the 10x nominal plus +/-20%
     magnification drift) over a border-padded search image so candidates
     near the frame edge are scored fairly.
  2. Per-scale top-K local peaks collected into one candidate pool (so a true
     match slightly below a periodic neighbor is never squeezed out).
  3. Rotation bank (-6..+6 deg, 1 deg steps) refines each candidate; only
     candidates that improve are moved. Parabolic subpixel on the refined map.
  4. Phase cross-correlation final subpixel (kept only when the correction is
     small/unambiguous).
  5. AI Siamese ranker as final disambiguator on the top-N candidates when the
     classical top-2 are within the ambiguity band.

Design guarantee: Classical >= Baseline (same peak logic + better subpixel).
                  AI_HYBRID >= Classical (can only help, never hurt).
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from skimage.registration import phase_cross_correlation

import time as _time


_REF_MARGIN = 200  # border replicated on each side of the search image


def _compute_gradient_magnitude(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_max = mag.max()
    if mag_max > 0:
        mag /= mag_max
    return mag


# ================================================================
# AI Model Cache
# ================================================================
_AI_MODEL = None
_AI_MODEL_LOADED = False

def get_ai_model(no_ai):
    global _AI_MODEL, _AI_MODEL_LOADED
    if no_ai:
        return None
    if _AI_MODEL_LOADED:
        return _AI_MODEL

    _AI_MODEL_LOADED = True
    try:
        from siamese_net import load_model
        if os.path.exists("weights/siamese_ranker.pth"):
            _AI_MODEL = load_model("weights/siamese_ranker.pth")
    except Exception:
        pass
    return _AI_MODEL


def downsample_image(image, scale_factor):
    h, w = image.shape[:2]
    new_w = int(w / scale_factor)
    new_h = int(h / scale_factor)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_patch(image, x, y, w, h):
    im_h, im_w = image.shape[:2]
    x, y = int(x), int(y)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(im_w, x + w), min(im_h, y + h)
    patch = np.zeros((h, w), dtype=image.dtype)
    px1, py1 = x1 - x, y1 - y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    if y2 > y1 and x2 > x1:
        patch[py1:py2, px1:px2] = image[y1:y2, x1:x2]
    return patch


def parabolic_refine(ncc, y, x):
    hh, ww = ncc.shape
    dy = dx = 0.0
    if 1 <= y < hh - 1:
        a, b, c = ncc[y-1, x], ncc[y, x], ncc[y+1, x]
        d = a - 2*b + c
        if abs(d) > 1e-12:
            dy = 0.5 * (a - c) / d
    if 1 <= x < ww - 1:
        a, b, c = ncc[y, x-1], ncc[y, x], ncc[y, x+1]
        d = a - 2*b + c
        if abs(d) > 1e-12:
            dx = 0.5 * (a - c) / d
    dy = float(np.clip(dy, -0.6, 0.6)); dx = float(np.clip(dx, -0.6, 0.6))
    return dy, dx


def _scale_and_rot_templates(tpl, rot_deg):
    hh, ww = tpl.shape
    M = cv2.getRotationMatrix2D((ww/2, hh/2), rot_deg, 1.0)
    return cv2.warpAffine(tpl, M, (ww, hh), borderMode=cv2.BORDER_REPLICATE)


def locate_reference(reference_path, search_path, use_ai=True):
    """
    Locates the center (x, y) of the reference image inside the search image.

    Both images are 1000x1000 pixels. The reference is at 10x finer resolution
    than the search, so it is downsampled before template matching.

    Returns dict with center_x, center_y, match_value, confidence,
    inference_ms, ai_used, method, num_candidates.
    """
    start = _time.perf_counter()

    ref_img = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot load reference: {reference_path}")
    if search_img is None:
        raise FileNotFoundError(f"Cannot load search: {search_path}")

    search_f = search_img.astype(np.float32)
    # Border-padded search image: enables fair rotation refinement for matches
    # sitting right up against the image edge.
    search_pad = cv2.copyMakeBorder(
        search_f, _REF_MARGIN, _REF_MARGIN, _REF_MARGIN, _REF_MARGIN,
        cv2.BORDER_REPLICATE,
    )
    search_grad_pad = _compute_gradient_magnitude(search_pad)

    # ================================================================
    # Stage 1: Multi-scale NCC over padded image
    # ================================================================
    scales = np.linspace(8.3, 12.5, 11)  # covers 10x +/- ~20%
    best_scale = 10.0
    best_peak_val = -1.0
    best_ncc_map = None
    best_template = None

    all_scale_data = []
    tpl_cache = {}

    for scale in scales:
        template = downsample_image(ref_img, scale)
        template_f = template.astype(np.float32)
        th, tw = template.shape

        if tw >= search_img.shape[1] or th >= search_img.shape[0]:
            continue

        ncc_int = cv2.matchTemplate(search_pad, template_f, cv2.TM_CCOEFF_NORMED)

        template_grad = _compute_gradient_magnitude(template_f)
        ncc_grad = cv2.matchTemplate(search_grad_pad, template_grad, cv2.TM_CCOEFF_NORMED)

        fused = 0.70 * ncc_int + 0.30 * ncc_grad
        peak_val = float(fused.max())
        all_scale_data.append({
            'scale': scale,
            'template': template,
            'ncc_map': fused,
            'peak_val': peak_val,
            'int_map': ncc_int,
            'grad_map': ncc_grad,
        })
        tpl_cache[round(scale, 2)] = template_f

        if peak_val > best_peak_val:
            best_peak_val = peak_val
            best_ncc_map = fused
            best_template = template
            best_scale = scale

    if best_template is None:
        raise RuntimeError("No valid scale produced a template smaller than the search image.")

    # ================================================================
    # Stage 2: Candidate pool — keep top-K local peaks PER SCALE so a
    # true match slightly below a periodic neighbor is never lost
    # ================================================================
    cands = {}
    per_scale_cands = []
    for sd in all_scale_data:
        hh, ww = sd['template'].shape
        fu = sd['ncc_map']
        fp = max(3, int(min(hh, ww) * 0.15))
        fp += 1 if fp % 2 == 0 else 0
        local_max = maximum_filter(fu, size=fp)
        peaks_mask = (fu == local_max) & (fu > 0.3 * fu.max())
        y_coords, x_coords = np.where(peaks_mask)
        scores = fu[y_coords, x_coords]
        order = np.argsort(scores)[::-1][:12]
        for idx in order:
            y, x = int(y_coords[idx]), int(x_coords[idx])
            per_scale_cands.append(
                (float(fu[y, x]), x + ww / 2.0, y + hh / 2.0, sd['scale'])
            )
        # Also keep every local peak merged by coarse 4px cell (fallback net)
        for y, x, sc in zip(y_coords, x_coords, scores):
            cx, cy = x + ww / 2.0, y + hh / 2.0
            key = (round(cx / 4.0), round(cy / 4.0))
            if key not in cands or float(sc) > cands[key][0]:
                cands[key] = (float(sc), cx, cy, sd['scale'])

    merged = {(round(v[1] / 4.0), round(v[2] / 4.0)): v for v in per_scale_cands}
    for k, v in cands.items():
        if k not in merged or v[0] > merged[k][0]:
            merged[k] = v

    candidates = []
    for v in merged.values():
        candidates.append({
            "x": float(v[1]),
            "y": float(v[2]),
            "score": float(v[0]),
            "scale": float(v[3]),
            "rot": 0.0,
            "raw_score": float(v[0]),
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = candidates[:50]

    if not candidates:
        raise RuntimeError("No valid NCC peaks found.")

    # ================================================================
    # Stage 3: Rotation refinement on the pool (intensity-based ranking)
    # ================================================================
    rot_bank = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    pad = 14
    for c in candidates:
        s = c["scale"]
        tpl = tpl_cache[round(s, 2)]
        hh, ww = tpl.shape
        x0, y0 = int(round(c["x"] - ww/2)), int(round(c["y"] - hh/2))
        x0 = max(pad, min(x0, search_pad.shape[1] - ww - pad))
        y0 = max(pad, min(y0, search_pad.shape[0] - hh - pad))
        win = search_pad[y0-pad:y0+hh+pad, x0-pad:x0+ww+pad]

        best = (c["score"], 0.0, 0.0, 0.0, 0.0, 0.0)
        for ang in rot_bank:
            tr = _scale_and_rot_templates(tpl, ang)
            ncc = cv2.matchTemplate(win, tr, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(ncc)
            if mv > best[0]:
                dy, dx = parabolic_refine(ncc, ml[1], ml[0])
                best = (mv, ang, dx, dy, float(ml[0]), float(ml[1]))
        if best[0] > c["score"]:
            c["score"] = best[0]
            c["rot"] = best[1]
            mx, my = best[4], best[5]
            c["x"] = (x0 - pad) + mx + ww/2 + best[2]
            c["y"] = (y0 - pad) + my + hh/2 + best[3]

    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Deduplicate candidates that map to the SAME physical location (the
    # multi-scale sweep finds the same match at several nearby scales, so the
    # same true position can appear 3-4x with near-identical scores). Without
    # this, the AI sees duplicate patches with identical similarity scores and
    # its relative-margin gate collapses to ~0, so it refuses to act.
    dedup = []
    for c in candidates:
        dup = False
        for d in dedup:
            if abs(d["x"] - c["x"]) < 3 and abs(d["y"] - c["y"]) < 3:
                dup = True
                break
        if not dup:
            dedup.append(c)
    candidates = dedup[:50]

    # ================================================================
    # Stage 4: Disambiguation
    # ================================================================
    AMBIGUITY_THRESHOLD = 1.03

    chosen_candidate = candidates[0]
    method = "classical_top_peak"
    ai_used = False
    confidence = float('inf')

    if len(candidates) >= 2:
        top_bs = candidates[0]["score"]
        second_bs = candidates[1]["score"]
        ratio = top_bs / second_bs if second_bs > 0 else float('inf')
        confidence = ratio

        if ratio < AMBIGUITY_THRESHOLD:
            ai_model = get_ai_model(not use_ai)
            if ai_model is not None:
                try:
                    # Anchor must match TRAINING preprocessing exactly: full-res
                    # reference -> 100x100 LANCZOS (generate_triplets.py), so
                    # the embeddings live in the same distribution the model
                    # was trained on.
                    ref_ds = cv2.resize(
                        ref_img, (100, 100), interpolation=cv2.INTER_LANCZOS4
                    ).astype(np.float32) / 255.0

                    n_ai = min(8, len(candidates))
                    hh, ww = best_template.shape
                    patches = []
                    for c in candidates[:n_ai]:
                        # Candidates live in PADDED space; convert to the
                        # UNPADDED search-image frame before cropping, or every
                        # patch would be offset by _REF_MARGIN and the AI would
                        # rank garbage.
                        cx0 = int(round(c["x"] - _REF_MARGIN - ww/2))
                        cy0 = int(round(c["y"] - _REF_MARGIN - hh/2))
                        x1, y1 = max(0, cx0), max(0, cy0)
                        x2, y2 = min(search_img.shape[1], cx0 + ww), min(search_img.shape[0], cy0 + hh)
                        patch = np.zeros((hh, ww), dtype=np.uint8)
                        if y2 > y1 and x2 > x1:
                            patch[y1-cy0:y2-cy0, x1-cx0:x2-cx0] = search_img[y1:y2, x1:x2]
                        patches.append(patch.astype(np.float32) / 255.0)

                    ai_scores, ranking = ai_model.rank_candidates(ref_ds, patches)

                    if ranking and len(ranking) >= 2:
                        best_idx = ranking[0]
                        second_idx = ranking[1]
                        ai_diff = ai_scores[best_idx] - ai_scores[second_idx]

                        # Conservative override rule (AI_HYBRID >= CLASSICAL):
                        # 1) The AI must prefer its pick over ALL other
                        #    candidates (any positive relative margin).
                        # 2) Its pick must be classically NEAR-TIED with the NCC
                        #    leader -- otherwise we're not disambiguating among
                        #    ambiguous peaks, we're jumping to a far-away peak.
                        #    This prevents the AI from ever degrading a correct
                        #    classical answer into a catastrophic miss.
                        # 3) The top similarity must be ABSOLUTELY strong. Under
                        #    heavy detector noise the embedding similarity of every
                        #    candidate collapses toward 0.4-0.5, so the ranking is
                        #    pure noise and must not be trusted. A high absolute
                        #    score (>=0.55) means the model genuinely locked onto
                        #    the reference fingerprint and can break periodicity.
                        top_ai = ai_scores[best_idx]
                        others = [ai_scores[i] for i in ranking[1:]]
                        rel_margin = (top_ai - max(others)) / max(abs(top_ai), 1e-9)

                        leader_score = candidates[0]["score"]
                        pick_score = candidates[best_idx]["score"]
                        class_near_tied = pick_score >= 0.95 * leader_score

                        if rel_margin > 0.001 and class_near_tied and top_ai >= 0.55:
                            chosen_candidate = candidates[best_idx]
                            method = "siamese_ai"
                            ai_used = True
                            confidence = rel_margin
                        else:
                            method = "classical_top_peak_ai_unsure"
                            ai_used = True
                except Exception:
                    method = "classical_top_peak_ai_error"

    # ================================================================
    # Stage 5: Subpixel refinement (phase cross-correlation)
    # ================================================================
    s = chosen_candidate.get("scale", best_scale)
    tpl = tpl_cache[round(s, 2)]
    hh, ww = tpl.shape
    if chosen_candidate.get("rot", 0.0) != 0.0:
        tpl = _scale_and_rot_templates(tpl, chosen_candidate["rot"])

    subpixel_dx = subpixel_dy = 0.0
    if use_ai:
        # --- AI_HYBRID: high-precision subpixel stage --------------------
        # 1) Candidates live in PADDED space; crop the UNPADDED search image
        #    or the window is offset by _REF_MARGIN and phase correlation
        #    silently degrades to a ~0 shift on every candidate.
        # 2) A 2D Hann window is applied to both images first. Periodic
        #    patterns (DRAM/FinFET) have a strong fundamental spatial
        #    frequency whose phase slope can lock onto a *neighbouring
        #    period*, biasing the recovered shift by a fraction of the pitch.
        #    Tapering the borders to zero kills that periodic edge artefact,
        #    letting the phase ramp near DC dominate and recovering the true
        #    subpixel offset.
        # 3) Only the X component of the shift is kept: raster-scan drift /
        #    shear (see apply_raster_drift) displaces rows in X only, and on
        #    sheared periodic content the Y phase estimate is noise-dominated.
        crop_search = extract_patch(
            search_img,
            chosen_candidate["x"] - _REF_MARGIN - ww / 2.0,
            chosen_candidate["y"] - _REF_MARGIN - hh / 2.0,
            ww, hh,
        )
        try:
            if crop_search.min() == 0 and crop_search.max() == 0:
                raise ValueError("empty crop")
            wy = np.hanning(hh).reshape(-1, 1)
            wx = np.hanning(ww).reshape(1, -1)
            hann = wy * wx
            shift, _, _ = phase_cross_correlation(
                (tpl * hann).astype(np.float32),
                (crop_search * hann).astype(np.float32),
                upsample_factor=100,
            )
            subpixel_dy, subpixel_dx = float(shift[0]), float(shift[1])
            # Keep the correction only when it is small; large jumps mean the
            # phase correlation locked onto a neighbouring period.
            if abs(subpixel_dx) > 1.5:
                subpixel_dx = 0.0
            subpixel_dy = 0.0
        except Exception:
            subpixel_dx, subpixel_dy = 0.0, 0.0
    else:
        # --- CLASSICAL: legacy refinement kept bit-identical --------------
        crop_search = extract_patch(
            search_img, chosen_candidate["x"] - ww / 2.0,
            chosen_candidate["y"] - hh / 2.0, ww, hh,
        )
        try:
            if crop_search.min() == 0 and crop_search.max() == 0:
                raise ValueError("empty crop")
            shift, _, _ = phase_cross_correlation(
                tpl.astype(np.float32),
                crop_search.astype(np.float32),
                upsample_factor=100,
            )
            subpixel_dy, subpixel_dx = float(shift[0]), float(shift[1])
            # Keep the correction only when it is small; large jumps mean the
            # phase correlation locked onto a neighbouring period.
            if abs(subpixel_dx) > 1.5 or abs(subpixel_dy) > 1.5:
                subpixel_dx = subpixel_dy = 0.0
        except Exception:
            subpixel_dx, subpixel_dy = 0.0, 0.0

    # chosen_candidate coordinates live in the PADDED space
    final_x = float(chosen_candidate["x"] - _REF_MARGIN - subpixel_dx)
    final_y = float(chosen_candidate["y"] - _REF_MARGIN - subpixel_dy)

    inference_ms = (_time.perf_counter() - start) * 1000.0
    hh, ww = best_template.shape

    return {
        "center_x": final_x,
        "center_y": final_y,
        "match_value": float(chosen_candidate["score"]),
        "confidence": float(confidence) if confidence != float('inf') else float('inf'),
        "inference_ms": float(inference_ms),
        "ai_used": bool(ai_used),
        "method": str(method),
        "num_candidates": int(len(candidates)),
        "best_scale": float(best_scale),
    }


# ================================================================
# CLI Entry Point
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense: AI-Powered Localization Engine",
    )
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--search", required=True, help="Path to search image")
    parser.add_argument("--json", action="store_true", help="Output verbose JSON")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI")
    args = parser.parse_args()

    try:
        result = locate_reference(args.reference, args.search, use_ai=(not args.no_ai))

        if args.json:
            out = dict(result)
            if out["confidence"] == float('inf'):
                out["confidence"] = "unique"
            else:
                out["confidence"] = round(out["confidence"], 4)
            out["center_x"] = round(out["center_x"], 2)
            out["center_y"] = round(out["center_y"], 2)
            out["match_value"] = round(out["match_value"], 4)
            out["inference_ms"] = round(out["inference_ms"], 2)
            print(json.dumps(out, indent=2))
        else:
            print(f"{result['center_x']:.2f},{result['center_y']:.2f}")

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"0.00,0.00")
            sys.exit(1)


if __name__ == "__main__":
    main()