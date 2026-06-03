"""
Entry point for `npm run dev`.

Runs the full pipeline with caching:
  - if retargeted.pkl is missing, do pose extraction + IK retargeting
  - then launch the side-by-side viewer (original video | Asimov sim)

Usage:
  python run.py                 # side-by-side (default)
  python run.py --mode sim      # MuJoCo viewer only
  python run.py --force         # ignore cache, re-extract + re-retarget
"""

import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VIDEO = os.path.join(HERE, "input_video.mp4")
POSE_PKL = os.path.join(HERE, "dance.pkl")
RETARGET_PKL = os.path.join(HERE, "retargeted.pkl")
MJCF = os.path.join(HERE, "..", "asimov-1", "sim-model", "xmls", "asimov.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sidebyside", "sim"], default="sidebyside")
    ap.add_argument("--force", action="store_true", help="re-extract + re-retarget")
    ap.add_argument("--save", default="result.mp4", help="output mp4 for side-by-side")
    args = ap.parse_args()

    if not os.path.exists(VIDEO):
        sys.exit(f"[error] {VIDEO} not found. Download a dance video first (see README).")

    # 1) + 2) pose extraction + retargeting (cached)
    if args.force or not os.path.exists(RETARGET_PKL):
        from extract_pose import extract_pose
        from retarget import retarget

        print("[1/2] Extracting pose with MediaPipe...")
        pose = extract_pose(VIDEO, POSE_PKL)

        print("[2/2] IK retargeting to Asimov...")
        r = retarget(pose, MJCF)
        with open(RETARGET_PKL, "wb") as f:
            pickle.dump(r, f)
    else:
        print(f"[cache] using {os.path.basename(RETARGET_PKL)} (use --force to rebuild)")
        with open(RETARGET_PKL, "rb") as f:
            r = pickle.load(f)

    # 3) playback
    if args.mode == "sim":
        from simulate import simulate
        simulate(r, MJCF)
    else:
        from visualize import render_side_by_side
        render_side_by_side(VIDEO, r, MJCF, output_path=args.save)


if __name__ == "__main__":
    main()
