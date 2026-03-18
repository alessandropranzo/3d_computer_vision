#!/usr/bin/env python3
"""
Extract frames from an iPhone video for CF-3DGS processing.

CF-3DGS expects:
  - JPEG images in {source_path}/images/*.jpg
  - Sorted filenames (alphabetical order = temporal order)
  - Max ~300 frames (auto-subsampled if exceeded)
  - If min(width, height) > 1000, the trainer halves the resolution internally

Usage:
    python extract_frames.py --video path/to/video.MOV
    python extract_frames.py --video path/to/video.MOV --fps 2 --resize 960
"""

import argparse
import os
import sys

import cv2
from PIL import Image


def extract_frames(video_path, output_dir, target_fps=None, max_frames=300,
                   resize_width=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        sys.exit(1)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / source_fps if source_fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {source_fps:.2f}")
    print(f"  Duration   : {duration:.1f}s  ({total_frames} frames)")

    if target_fps is None:
        target_fps = source_fps

    frame_interval = max(1, round(source_fps / target_fps))
    estimated_output = total_frames // frame_interval

    if estimated_output > max_frames:
        frame_interval = max(1, total_frames // max_frames)
        print(f"  Adjusted interval to {frame_interval} to stay under "
              f"{max_frames} frames")

    os.makedirs(output_dir, exist_ok=True)

    if resize_width is not None:
        aspect = height / width
        resize_height = int(resize_width * aspect)
        resize_height = resize_height - (resize_height % 2)
        resize_dim = (resize_width, resize_height)
        print(f"  Resizing to: {resize_width}x{resize_height}")
    else:
        resize_dim = None

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if resize_dim is not None:
                frame = cv2.resize(frame, resize_dim,
                                   interpolation=cv2.INTER_LANCZOS4)
            out_path = os.path.join(output_dir, f"{saved:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"\nExtracted {saved} frames to '{output_dir}'")

    if saved > 0:
        sample = Image.open(os.path.join(output_dir, "00000.jpg"))
        w, h = sample.size
        print(f"  Output resolution: {w}x{h}")
        if min(w, h) > 1000:
            print("  Note: CF-3DGS will internally halve this resolution "
                  "during training.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from iPhone video for CF-3DGS")
    parser.add_argument("--video", "-v", required=True,
                        help="Path to the input video (.MOV, .mp4)")
    parser.add_argument("--output", "-o",
                        default="data/my_room/images",
                        help="Output directory for frames "
                             "(default: data/my_room/images)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Target FPS for extraction. "
                             "Default: use all frames from the video. "
                             "Recommended: 2-5 for a 30fps video.")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Maximum number of frames to extract "
                             "(default: 300, matches CF-3DGS limit)")
    parser.add_argument("--resize", type=int, default=None,
                        help="Resize frames to this width "
                             "(maintains aspect ratio). "
                             "Recommended: 960 for 1080p input.")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps, args.max_frames,
                   args.resize)


if __name__ == "__main__":
    main()
