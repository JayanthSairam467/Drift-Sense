import os
import json
import time
import math
import argparse
import glob
from collections import defaultdict

import cv2
import scipy.stats

from localize import locate_reference


def baseline_predict(ref_path, search_path):
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or search is None:
        return 0, 0
    
    # Downsample reference by 10x
    template = cv2.resize(ref, (ref.shape[1] // 10, ref.shape[0] // 10), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    cx = max_loc[0] + template.shape[1] / 2.0
    cy = max_loc[1] + template.shape[0] / 2.0
    return cx, cy


def calculate_error(pred_x, pred_y, gt_x, gt_y):
    return math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark localization pipeline")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--num-pairs", type=int, default=0, help="Number of pairs to test (0=all)")
    args = parser.parse_args()

    dataset_path = args.dataset
    pair_dirs = glob.glob(os.path.join(dataset_path, "pair_*"))
    
    # Filter directories only just in case
    pair_dirs = [d for d in pair_dirs if os.path.isdir(d)]
    pair_dirs.sort()

    if args.num_pairs > 0:
        pair_dirs = pair_dirs[:args.num_pairs]

    results = []
    
    print(f"Benchmarking {len(pair_dirs)} pairs...")
    print(f"{'Pair':<15} | {'Difficulty':<10} | {'Base (px)':<10} | {'Class (px)':<10} | {'AI (px)':<10}")
    print("-" * 65)

    stats = {
        "BASELINE": {"errors": [], "times": []},
        "CLASSICAL": {"errors": [], "times": []},
        "AI_HYBRID": {"errors": [], "times": [], "ai_used": 0}
    }

    # For McNemar's test (correct if error <= 5px)
    ai_correct = []
    base_correct = []

    for pdir in pair_dirs:
        pair_name = os.path.basename(pdir)
        ref_path = os.path.join(pdir, "reference.png")
        search_path = os.path.join(pdir, "search.png")
        gt_path = os.path.join(pdir, "ground_truth.json")

        if not (os.path.exists(ref_path) and os.path.exists(search_path) and os.path.exists(gt_path)):
            continue

        with open(gt_path, 'r') as f:
            gt = json.load(f)
        
        gt_x = gt.get("center_x", 0)
        gt_y = gt.get("center_y", 0)
        difficulty = gt.get("difficulty", "unknown")

        # Baseline
        t0 = time.time()
        b_x, b_y = baseline_predict(ref_path, search_path)
        t_base = (time.time() - t0) * 1000
        e_base = calculate_error(b_x, b_y, gt_x, gt_y)
        
        # Classical
        t0 = time.time()
        res_c = locate_reference(ref_path, search_path, use_ai=False)
        t_class = (time.time() - t0) * 1000
        e_class = calculate_error(res_c["center_x"], res_c["center_y"], gt_x, gt_y)

        # AI_HYBRID
        t0 = time.time()
        res_a = locate_reference(ref_path, search_path, use_ai=True)
        t_ai = (time.time() - t0) * 1000
        e_ai = calculate_error(res_a["center_x"], res_a["center_y"], gt_x, gt_y)

        print(f"{pair_name:<15} | {difficulty:<10} | {e_base:<10.2f} | {e_class:<10.2f} | {e_ai:<10.2f}")

        stats["BASELINE"]["errors"].append(e_base)
        stats["BASELINE"]["times"].append(t_base)
        
        stats["CLASSICAL"]["errors"].append(e_class)
        stats["CLASSICAL"]["times"].append(t_class)
        
        stats["AI_HYBRID"]["errors"].append(e_ai)
        stats["AI_HYBRID"]["times"].append(t_ai)
        if res_a.get("ai_used", False):
            stats["AI_HYBRID"]["ai_used"] += 1

        base_corr = e_base <= 5.0
        ai_corr = e_ai <= 5.0
        base_correct.append(base_corr)
        ai_correct.append(ai_corr)

        results.append({
            "pair": pair_name,
            "difficulty": difficulty,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "baseline": {"x": b_x, "y": b_y, "error": e_base, "time_ms": t_base},
            "classical": {"x": res_c["center_x"], "y": res_c["center_y"], "error": e_class, "time_ms": t_class},
            "ai_hybrid": {"x": res_a["center_x"], "y": res_a["center_y"], "error": e_ai, "time_ms": t_ai, "ai_used": res_a.get("ai_used", False)}
        })

    print("\n--- Summary Report ---")
    methods = ["BASELINE", "CLASSICAL", "AI_HYBRID"]
    
    print(f"{'Method':<12} | {'Avg Err':<8} | {'Med Err':<8} | {'Avg Time':<8} | {'<=1px':<6} | {'<=2px':<6} | {'<=5px':<6} | {'<=10px':<6}")
    print("-" * 80)
    for m in methods:
        errs = stats[m]["errors"]
        times = stats[m]["times"]
        if not errs:
            continue
        errs_sorted = sorted(errs)
        avg_e = sum(errs) / len(errs)
        med_e = errs_sorted[len(errs)//2]
        avg_t = sum(times) / len(times)
        
        le1 = sum(1 for e in errs if e <= 1.0) / len(errs) * 100
        le2 = sum(1 for e in errs if e <= 2.0) / len(errs) * 100
        le5 = sum(1 for e in errs if e <= 5.0) / len(errs) * 100
        le10 = sum(1 for e in errs if e <= 10.0) / len(errs) * 100
        
        print(f"{m:<12} | {avg_e:<8.2f} | {med_e:<8.2f} | {avg_t:<8.1f} | {le1:<5.1f}% | {le2:<5.1f}% | {le5:<5.1f}% | {le10:<5.1f}%")

    # By-difficulty breakdown
    print("\n--- By-Difficulty Breakdown (Mean Error) ---")
    diffs = defaultdict(lambda: {"BASELINE": [], "CLASSICAL": [], "AI_HYBRID": []})
    for r in results:
        diff_level = r["difficulty"]
        diffs[diff_level]["BASELINE"].append(r["baseline"]["error"])
        diffs[diff_level]["CLASSICAL"].append(r["classical"]["error"])
        diffs[diff_level]["AI_HYBRID"].append(r["ai_hybrid"]["error"])
    
    for diff_level, data in diffs.items():
        avg_base = sum(data["BASELINE"]) / len(data["BASELINE"]) if data["BASELINE"] else 0
        avg_class = sum(data["CLASSICAL"]) / len(data["CLASSICAL"]) if data["CLASSICAL"] else 0
        avg_ai = sum(data["AI_HYBRID"]) / len(data["AI_HYBRID"]) if data["AI_HYBRID"] else 0
        print(f"{diff_level:<15} | Base: {avg_base:<7.2f} | Class: {avg_class:<7.2f} | AI: {avg_ai:<7.2f}")

    print("\n--- McNemar's Test (AI_HYBRID vs BASELINE, <=5px correct) ---")
    both_correct = 0
    ai_only = 0
    base_only = 0
    both_wrong = 0
    for ac, bc in zip(ai_correct, base_correct):
        if ac and bc: both_correct += 1
        elif ac and not bc: ai_only += 1
        elif not ac and bc: base_only += 1
        else: both_wrong += 1
        
    print(f"Both correct: {both_correct}")
    print(f"AI only correct: {ai_only}")
    print(f"Baseline only correct: {base_only}")
    print(f"Both wrong: {both_wrong}")
    
    denom = ai_only + base_only
    if denom > 0:
        chi2 = ((ai_only - base_only) ** 2) / denom
        p_val = scipy.stats.chi2.sf(chi2, 1)
        print(f"Chi2 statistic: {chi2:.4f}, p-value: {p_val:.4e}")
    else:
        print("Chi2 statistic: 0.0000, p-value: 1.0000e+00 (No discordant pairs)")

    print(f"\nAI used in {stats['AI_HYBRID']['ai_used']} out of {len(results)} pairs.")

    out_path = os.path.join(dataset_path, "benchmark_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
