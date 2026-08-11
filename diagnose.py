"""Quick diagnostic script for localization accuracy."""
from localize import locate_reference
import json
import math

# Test on pair_0000
r = locate_reference('dataset/reference/ref_0000.png', 'dataset/search/search_0000.png', use_ai=False)
gt_x, gt_y = 120, 96
err = math.sqrt((float(r["center_x"]) - gt_x)**2 + (float(r["center_y"]) - gt_y)**2)

print(f"Predicted: ({float(r['center_x']):.1f}, {float(r['center_y']):.1f})")
print(f"Ground Truth: ({gt_x}, {gt_y})")
print(f"Error: {err:.1f} px")
print(f"Method: {r['method']}")
print(f"Candidates: {r['num_candidates']}")
print(f"Match Value: {float(r['match_value']):.4f}")

# Test on pair_0039 (which got 0.10 error - the good one)
gt_x2, gt_y2 = json.load(open('dataset/metadata.json'))['pair_0039']['gt_x'], json.load(open('dataset/metadata.json'))['pair_0039']['gt_y']
r2 = locate_reference('dataset/reference/ref_0039.png', 'dataset/search/search_0039.png', use_ai=False)
err2 = math.sqrt((float(r2["center_x"]) - gt_x2)**2 + (float(r2["center_y"]) - gt_y2)**2)
print(f"\nPair 0039:")
print(f"Predicted: ({float(r2['center_x']):.1f}, {float(r2['center_y']):.1f})")
print(f"Ground Truth: ({gt_x2}, {gt_y2})")
print(f"Error: {err2:.1f} px")
print(f"Method: {r2['method']}")

# Show the NCC heatmap peaks for pair_0000 to understand ambiguity
import cv2
import numpy as np
ref = cv2.imread('dataset/reference/ref_0000.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
search = cv2.imread('dataset/search/search_0000.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
result = cv2.matchTemplate(search, ref, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(f"\nNCC Heatmap for pair_0000:")
print(f"Max NCC: {max_val:.4f} at {max_loc}")
print(f"NCC at GT location (top-left={gt_x-50},{gt_y-50}): ", end="")
gt_tl_x = gt_x - 50  # center to top-left
gt_tl_y = gt_y - 50
if 0 <= gt_tl_y < result.shape[0] and 0 <= gt_tl_x < result.shape[1]:
    print(f"{result[gt_tl_y, gt_tl_x]:.4f}")
else:
    print("OUT OF BOUNDS")

# Count peaks above 95% of max
threshold = max_val * 0.95
above = np.sum(result >= threshold)
print(f"Pixels above 95% threshold: {above}")
print(f"Threshold value: {threshold:.4f}")
