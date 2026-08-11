"""
Drift-Sense: Inference Script (Hackathon Submission)
=====================================================
THIS IS THE SCRIPT APPLIED MATERIALS WILL RUN ON TEST DATA.

Usage:
    python inference.py --reference path/to/reference.png --search path/to/search.png

Output:
    Prints a single JSON object with predicted center (x, y).
    {
      "center_x": 512.34,
      "center_y": 487.21
    }

Requirements:
  - Must accept reference image path and search image path as arguments
  - Must output a single (x, y) coordinate
  - Must run without manual edits
  - Must load model weights automatically if available
  - Must gracefully fall back to classical method if model is missing
"""

import sys
import json
from localize import locate_reference


def main():
    # Parse simple arguments
    args = sys.argv[1:]
    ref_path = None
    search_path = None

    i = 0
    while i < len(args):
        if args[i] in ("--reference", "-r") and i + 1 < len(args):
            ref_path = args[i + 1]
            i += 2
        elif args[i] in ("--search", "-s") and i + 1 < len(args):
            search_path = args[i + 1]
            i += 2
        else:
            i += 1

    if ref_path is None or search_path is None:
        print(json.dumps({"error": "Usage: python inference.py --reference ref.png --search search.png"}))
        sys.exit(1)

    try:
        # FIX: locate_reference() returns a dict (center_x, center_y,
        # match_value, confidence, inference_ms, ai_used, method,
        # num_candidates -- 8 keys), not a 6-item tuple. The previous
        # version did `x, y, _, _, _, _ = locate_reference(...)`, which
        # unpacks a dict's KEYS (not values) by iteration and crashes with
        # "too many values to unpack (expected 6)" on every single call --
        # this is the script Applied Materials runs directly, so this bug
        # alone would have scored zero regardless of algorithm quality.
        result = locate_reference(ref_path, search_path)
        output = {
            "center_x": round(float(result["center_x"]), 2),
            "center_y": round(float(result["center_y"]), 2)
        }
        print(json.dumps(output))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
