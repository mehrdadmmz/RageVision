"""Extract frames from video files for dataset creation."""

import argparse
import os

from ragevision.data.frame_extractor import extract_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from video files")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--max-frames", type=int, default=90, help="Max frames to extract per video")
    parser.add_argument("--prefix", type=str, default="frame", help="Filename prefix (e.g. 'rage', 'non_rage')")
    parser.add_argument("--ext", type=str, default=".mov", help="Video file extension to look for")
    return parser.parse_args()


def main():
    args = parse_args()

    video_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(args.ext)
    ])

    if not video_files:
        print(f"No {args.ext} files found in {args.input_dir}")
        return

    print(f"Found {len(video_files)} video(s) in {args.input_dir}")

    current_index = 0
    total_frames = 0

    for video_path in video_files:
        print(f"Processing: {os.path.basename(video_path)}")
        count = extract_frames(
            video_path=video_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            start_index=current_index,
            prefix=args.prefix,
        )
        print(f"  Extracted {count} frames (index {current_index + 1}-{current_index + count})")
        current_index += count
        total_frames += count

    print(f"\nDone. Extracted {total_frames} total frames to {args.output_dir}")


if __name__ == "__main__":
    main()
