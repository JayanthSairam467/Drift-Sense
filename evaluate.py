#!/usr/bin/env python3
"""Drift-Sense: Single-pair evaluation script.

Usage:
    python evaluate.py --reference ref.png --search search.png
    python evaluate.py --reference ref.png --search search.png --gt-x 500.0 --gt-y 500.0
"""

import argparse
import numpy as np
from localize import locate_reference

def main():
    parser = argparse.ArgumentParser(description="Evaluate localization on a single image pair.")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--search", required=True, help="Path to search image")
    parser.add_argument("--gt-x", type=float, default=None, help="Ground truth X coordinate")
    parser.add_argument("--gt-y", type=float, default=None, help="Ground truth Y coordinate")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI (classical only)")
    args = parser.parse_args()

    result = locate_reference(args.reference, args.search, use_ai=(not args.no_ai))
    
    print(f"Predicted: ({result['center_x']:.2f}, {result['center_y']:.2f})")
    print(f"Method:    {result['method']}")
    print(f"AI Used:   {result['ai_used']}")
    print(f"Time:      {result['inference_ms']:.1f} ms")
    
    if args.gt_x is not None and args.gt_y is not None:
        error = np.sqrt((result['center_x'] - args.gt_x)**2 + (result['center_y'] - args.gt_y)**2)
        print(f"\nGround Truth: ({args.gt_x:.2f}, {args.gt_y:.2f})")
        print(f"Error:        {error:.2f} px")
        print(f"Status:       {'PASS' if error <= 5.0 else 'FAIL'} (threshold: 5.0 px)")

if __name__ == "__main__":
    main()
