import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import mediapy as media
from pathlib import Path
from typing import Generator, List
from tqdm import tqdm


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_model():
    print("Loading FILM from TensorFlow Hub...")
    model = hub.load("https://tfhub.dev/google/film/1")
    print("Model loaded.")
    return model


# ──────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load image as float32 in [0, 1] with batch dimension."""
    img = tf.image.decode_image(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()[np.newaxis, ...]   # (1, H, W, 3)


def save_image(img: np.ndarray, path: str):
    """Save float32 [0,1] image to disk."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    tf.io.write_file(path, tf.image.encode_png(img_uint8))


# ──────────────────────────────────────────────
# Recursive interpolation
# ──────────────────────────────────────────────

def _recursive_generator(
    model,
    frame1: np.ndarray,
    frame2: np.ndarray,
    num_recursions: int,
    bar: tqdm = None,
) -> Generator[np.ndarray, None, None]:
    """
    Recursively generate interpolated frames between frame1 and frame2.
    Yields frames in temporal order (excluding frame1, including frame2).
    Each recursion doubles the number of in-between frames:
      num_recursions=1 → 1 mid frame
      num_recursions=2 → 3 frames
      num_recursions=n → 2^n - 1 frames
    """
    if num_recursions == 0:
        yield frame2
        return

    # Run model at t=0.5
    inputs = {
        "x0": frame1,
        "x1": frame2,
        "time": np.array([[0.5]], dtype=np.float32),
    }
    mid = model(inputs)["image"].numpy()

    # Recurse: left half, then right half
    yield from _recursive_generator(model, frame1, mid,  num_recursions - 1, bar)
    yield from _recursive_generator(model, mid,  frame2, num_recursions - 1, bar)

    if bar:
        bar.update(1)


def interpolate_pair(
    model,
    frame1: np.ndarray,
    frame2: np.ndarray,
    times_to_interpolate: int,
) -> List[np.ndarray]:
    """
    Return all frames between frame1 and frame2 (inclusive of frame2,
    exclusive of frame1), in temporal order.
    """
    n_mid = 2 ** times_to_interpolate - 1
    with tqdm(total=n_mid, desc="  interpolating pair", leave=False) as bar:
        frames = list(_recursive_generator(model, frame1, frame2, times_to_interpolate, bar))
    return frames


# ──────────────────────────────────────────────
# Batch interpolation over a directory
# ──────────────────────────────────────────────

def interpolate_frames(
    input_dir: str,
    output_dir: str,
    times_to_interpolate: int = 3,
    fps: int = 30,
    save_frames: bool = True,
    save_video: bool = True,
):
    """
    Interpolate all frames in input_dir and write results to output_dir.

    Args:
        times_to_interpolate:  Recursion depth.
                               1 → 1 mid frame per pair (2× frame rate)
                               2 → 3 mid frames per pair (4× frame rate)
                               3 → 7 mid frames per pair (8× frame rate)  ← default
        fps:                   Output video frame rate.
        save_frames:           Write individual .png frames to output_dir/frames/.
        save_video:            Write output_video.mp4 to output_dir/.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    frames_path = output_path / "frames"

    output_path.mkdir(parents=True, exist_ok=True)
    if save_frames:
        frames_path.mkdir(parents=True, exist_ok=True)

    # Collect and sort input frames
    exts = {".png", ".jpg", ".jpeg"}
    input_frames = sorted(
        [p for p in input_path.iterdir() if p.suffix.lower() in exts]
    )
    if len(input_frames) < 2:
        raise ValueError(f"Need at least 2 frames in {input_dir}, found {len(input_frames)}.")

    print(f"Found {len(input_frames)} input frames.")
    n_out = (2 ** times_to_interpolate + 1) * (len(input_frames) - 1)
    print(f"times_to_interpolate={times_to_interpolate} → {n_out} output frames at {fps} fps "
          f"({n_out / fps:.1f}s video).")

    model = load_model()

    all_frames = []
    frame_idx  = 0

    for i in range(len(input_frames) - 1):
        print(f"[{i+1}/{len(input_frames)-1}] {input_frames[i].name} → {input_frames[i+1].name}")

        f1 = load_image(str(input_frames[i]))
        f2 = load_image(str(input_frames[i + 1]))

        # Always include the first frame of each pair
        if i == 0:
            if save_frames:
                save_image(f1[0], str(frames_path / f"frame_{frame_idx:06d}.png"))
            all_frames.append(f1[0])
            frame_idx += 1

        # Generate and save interpolated + ending frames
        interp = interpolate_pair(model, f1, f2, times_to_interpolate)
        for frame in interp:
            if save_frames:
                save_image(frame[0], str(frames_path / f"frame_{frame_idx:06d}.png"))
            all_frames.append(frame[0])
            frame_idx += 1

    print(f"Generated {len(all_frames)} frames total.")

    if save_video:
        video_path = str(output_path / "output_video.mp4")
        print(f"Writing video to {video_path} ...")
        media.write_video(video_path, np.stack(all_frames), fps=fps)
        print(f"Video saved.")

    if save_frames:
        print(f"Individual frames saved to {frames_path}")

    return all_frames


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interpolate a directory of frames using Google FILM."
    )
    parser.add_argument("--input_dir",            type=str, required=True,
                        help="Directory of input frames (sorted alphabetically)")
    parser.add_argument("--output_dir",           type=str, required=True,
                        help="Output directory for interpolated frames and video")
    parser.add_argument("--times_to_interpolate", type=int, default=3,
                        help="Recursion depth. 1=2x, 2=4x, 3=8x frame rate (default: 3)")
    parser.add_argument("--fps",                  type=int, default=30,
                        help="Output video FPS (default: 30)")
    parser.add_argument("--no_frames",            action="store_true",
                        help="Skip saving individual frame PNGs")
    parser.add_argument("--no_video",             action="store_true",
                        help="Skip saving output video")
    args = parser.parse_args()

    interpolate_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        times_to_interpolate=args.times_to_interpolate,
        fps=args.fps,
        save_frames=not args.no_frames,
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    import os

    # Must be set before importing TensorFlow
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"       # suppress oneDNN warnings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # explicitly expose GPU 0 to TF

    import tensorflow as tf

    # Configure GPU memory growth — prevents TF from grabbing all 20GB at once
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {[g.name for g in gpus]}")
    else:
        print("WARNING: No GPU found by TensorFlow — running on CPU.")
    main()