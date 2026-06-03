import argparse
import pickle
import sys
from pathlib import Path

MJCF_PATH = str(Path(__file__).parent.parent / "asimov-1" / "sim-model" / "xmls" / "asimov.xml")


def main():
    parser = argparse.ArgumentParser(
        description="K-pop dance → Asimov MuJoCo simulation pipeline"
    )
    parser.add_argument("--video", required=True, help="YouTube URL or local mp4 path")
    parser.add_argument("--output", default="dance.pkl", help="Output pose file (default: dance.pkl)")
    parser.add_argument("--song", default="", help="Song name label (optional)")
    parser.add_argument("--mode", choices=["sim", "sidebyside", "both"], default="sidebyside",
                        help="sim=MuJoCo only, sidebyside=video+sim, both=run both")
    parser.add_argument("--save-video", default=None, help="Save side-by-side output to mp4")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction, load existing --output pkl")
    args = parser.parse_args()

    pose_pkl = args.output
    video_path = args.video

    # ── Step 1 & 2: Download + Extract ─────────────────────────────────────
    if not args.skip_extract:
        from extract_pose import download_video, extract_pose

        # Download if URL
        if video_path.startswith("http"):
            local_mp4 = "input_video.mp4"
            download_video(video_path, local_mp4)
            video_path = local_mp4

        print(f"\n{'='*50}")
        print(f"  Song: {args.song or 'unknown'}")
        print(f"  Video: {video_path}")
        print(f"{'='*50}\n")

        pose_data = extract_pose(video_path, pose_pkl)
    else:
        print(f"Loading existing pose data from {pose_pkl}...")
        with open(pose_pkl, "rb") as f:
            pose_data = pickle.load(f)

    # ── Step 3: Retarget ────────────────────────────────────────────────────
    print("\nRetargeting to Asimov joints...")
    from retarget import retarget
    retargeted = retarget(pose_data, MJCF_PATH)

    n_frames = len(retargeted["waist_yaw_joint"])
    fps = retargeted["fps"]
    print(f"Retargeted {n_frames} frames ({n_frames/fps:.1f}s) to Asimov joints")

    # ── Step 4 & 5: Simulate / Visualize ───────────────────────────────────
    if args.mode in ("sim", "both"):
        print("\nLaunching MuJoCo simulation...")
        from simulate import simulate
        simulate(retargeted, MJCF_PATH)

    if args.mode in ("sidebyside", "both"):
        if video_path.startswith("http"):
            video_path = "input_video.mp4"
        print("\nLaunching side-by-side visualization...")
        from visualize import render_side_by_side
        render_side_by_side(video_path, retargeted, MJCF_PATH, output_path=args.save_video)


if __name__ == "__main__":
    main()
