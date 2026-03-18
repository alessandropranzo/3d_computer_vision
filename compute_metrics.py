import os
import sys
import argparse
import numpy as np
import torch
import cv2
import lpips
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import VGGT utilities (same as extract_visualizations.py)
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Reuse generate_predictions from extract_visualizations.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_visualizations import generate_predictions


# ──────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────

def load_image_rgb(path: str) -> np.ndarray:
    """Load image as float32 numpy array in [0, 1], shape (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_depth(path: str) -> np.ndarray:
    """
    Load a depth map. Supports:
      - 16-bit PNG (values in mm or raw uint16)
      - 32-bit float EXR / .npy
      - Single-channel 8-bit PNG (normalized)
    Returns float32 numpy array (H, W).
    """
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)

    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load depth: {path}")
    if len(img.shape) == 3:
        img = img[..., 0]   # take first channel if multi-channel
    return img.astype(np.float32)


def to_tensor(img: np.ndarray, device="cuda") -> torch.Tensor:
    """Convert (H, W, 3) float32 numpy in [0,1] to (1, 3, H, W) tensor in [-1, 1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    t = t * 2.0 - 1.0      # LPIPS expects [-1, 1]
    return t.to(device)


def resize_to_match(img: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize img to match target's (H, W) if they differ."""
    if img.shape[:2] != target.shape[:2]:
        h, w = target.shape[:2]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


# ──────────────────────────────────────────────
# VGGT depth extraction
# ──────────────────────────────────────────────

def get_vggt_depth_maps(image_paths: list, device="cuda") -> list:
    """
    Extract per-frame depth maps from VGGT's dedicated depth head.
    Returns a list of (H, W) float32 numpy arrays, one per input frame.
    """
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    print(f"  [VGGT depth] Loading {len(image_paths)} images...")
    images = load_and_preprocess_images(image_paths).to(device)  # (S, C, H, W)

    print("  [VGGT depth] Running inference...")
    predictions = generate_predictions(images, device=device, dtype=dtype)

    if "depth" not in predictions:
        raise RuntimeError(
            "'depth' not in VGGT predictions. "
            "Make sure your generate_predictions() call includes the depth head. "
            "Fall back to manual world_points projection if depth head is unavailable."
        )

    depth = predictions["depth"].cpu().float()  # (S, H, W) or (1, S, H, W)

    if depth.ndim == 4:
        depth = depth.squeeze(0)                # → (S, H, W)

    depth_maps = []
    for i in range(depth.shape[0]):
        d = depth[i].numpy()                    # (H, W)
        # VGGT depth head outputs positive values — but guard anyway
        if d.mean() < 0:
            d = -d
        depth_maps.append(d.astype(np.float32))

    print(f"  [VGGT depth] Extracted {len(depth_maps)} depth maps "
          f"(range [{np.min(depth_maps):.3f}, {np.max(depth_maps):.3f}])")
    return depth_maps


# ──────────────────────────────────────────────
# RGB metrics
# ──────────────────────────────────────────────

def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return psnr(gt, pred, data_range=1.0)


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return ssim(gt, pred, data_range=1.0, channel_axis=2)


def compute_lpips(pred: np.ndarray, gt: np.ndarray, loss_fn, device="cuda") -> float:
    t_pred = to_tensor(pred, device)
    t_gt   = to_tensor(gt,   device)
    with torch.no_grad():
        dist = loss_fn(t_pred, t_gt)
    return dist.item()


# ──────────────────────────────────────────────
# Depth metrics
# ──────────────────────────────────────────────

def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, min_depth=1e-3, max_depth=80.0):
    """
    Standard depth estimation metrics, following Eigen et al.

    Returned metrics:
      abs_rel:   Mean absolute relative error         ↓ lower is better
      sq_rel:    Mean squared relative error           ↓ lower is better
      rmse:      Root mean squared error               ↓ lower is better
      rmse_log:  RMSE in log space                     ↓ lower is better
      a1:        % pixels with ratio < 1.25            ↑ higher is better
      a2:        % pixels with ratio < 1.25²           ↑ higher is better
      a3:        % pixels with ratio < 1.25³           ↑ higher is better
    """
    pred_depth = resize_to_match(pred_depth, gt_depth)

    # Mask out invalid (zero / out-of-range) depth values
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) \
          & (pred_depth > min_depth) & (pred_depth < max_depth)

    if valid.sum() == 0:
        raise ValueError("No valid depth pixels found — check depth range / units.")

    pred = pred_depth[valid]
    gt   = gt_depth[valid]

    # Scale-invariant alignment (median scaling) — important when
    # VGGT depth is in arbitrary units vs. metric GT depth
    scale = np.median(gt) / np.median(pred)
    pred  = pred * scale

    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel  = np.mean(np.abs(gt - pred) / gt)
    sq_rel   = np.mean(((gt - pred) ** 2) / gt)
    rmse     = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    return {
        "abs_rel":  abs_rel,
        "sq_rel":   sq_rel,
        "rmse":     rmse,
        "rmse_log": rmse_log,
        "a1":       a1,
        "a2":       a2,
        "a3":       a3,
    }


# ──────────────────────────────────────────────
# Per-frame evaluation
# ──────────────────────────────────────────────

def evaluate_frame(pred_rgb_path, gt_rgb_path, loss_fn, device,
                   pred_depth: np.ndarray = None,
                   gt_depth_path=None,
                   min_depth=1e-3, max_depth=80.0):
    """
    Evaluate a single frame.

    pred_depth: (H, W) float32 depth array already produced by VGGT
                (or None to skip depth metrics).
    gt_depth_path: path to ground-truth depth file on disk.
    """
    pred_rgb = load_image_rgb(pred_rgb_path)
    gt_rgb   = load_image_rgb(gt_rgb_path)
    pred_rgb = resize_to_match(pred_rgb, gt_rgb)

    results = {
        "psnr":  compute_psnr(pred_rgb, gt_rgb),
        "ssim":  compute_ssim(pred_rgb, gt_rgb),
        "lpips": compute_lpips(pred_rgb, gt_rgb, loss_fn, device),
    }

    if pred_depth is not None and gt_depth_path is not None:
        gt_depth = load_depth(gt_depth_path)
        depth_metrics = compute_depth_metrics(pred_depth, gt_depth,
                                              min_depth, max_depth)
        results.update(depth_metrics)

    return results


# ──────────────────────────────────────────────
# Directory-level evaluation
# ──────────────────────────────────────────────

def evaluate_directory(
    pred_dir,
    gt_dir,
    gt_depth_dir=None,
    use_vggt_depth=False,
    device="cuda",
    min_depth=1e-3,
    max_depth=80.0,
    depth_ext=".png",
):
    """
    Evaluate all frame pairs in pred_dir vs gt_dir.

    If use_vggt_depth=True, VGGT is run on the predicted frames and the
    resulting depth maps are used instead of loading depth files from disk.
    gt_depth_dir is still required for the ground-truth depth side.
    """
    exts = {".png", ".jpg", ".jpeg"}
    pred_frames = sorted([p for p in Path(pred_dir).iterdir() if p.suffix.lower() in exts])
    gt_frames   = sorted([p for p in Path(gt_dir).iterdir()   if p.suffix.lower() in exts])

    if len(pred_frames) != len(gt_frames):
        print(f"WARNING: pred has {len(pred_frames)} frames, gt has {len(gt_frames)}. "
              f"Matching by filename.")
        gt_map = {p.name: p for p in gt_frames}
        pairs  = [(p, gt_map[p.name]) for p in pred_frames if p.name in gt_map]
    else:
        pairs = list(zip(pred_frames, gt_frames))

    print(f"Evaluating {len(pairs)} frame pairs…")

    # ── Optionally generate VGGT depth maps for all predicted frames ─────────
    vggt_depths = None
    if use_vggt_depth:
        pred_paths_ordered = [str(p) for p, _ in pairs]
        print("Generating VGGT depth maps for predicted frames…")
        vggt_depths = get_vggt_depth_maps(pred_paths_ordered, device=device)

    # ── LPIPS model ──────────────────────────────────────────────────────────
    loss_fn = lpips.LPIPS(net="alex").to(device)

    all_results = []

    for i, (pred_path, gt_path) in enumerate(pairs):
        stem = pred_path.stem

        # Depth for this frame
        pred_depth_arr = vggt_depths[i] if vggt_depths is not None else None
        gt_d = Path(gt_depth_dir) / (stem + depth_ext) if gt_depth_dir else None
        gt_d = gt_d if (gt_d and gt_d.exists()) else None

        # Only compute depth metrics if we have both sides
        has_depth = (pred_depth_arr is not None) and (gt_d is not None)

        try:
            metrics = evaluate_frame(
                pred_path, gt_path, loss_fn, device,
                pred_depth=pred_depth_arr if has_depth else None,
                gt_depth_path=gt_d        if has_depth else None,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            metrics["frame"] = pred_path.name
            all_results.append(metrics)

            # Per-frame print
            depth_str = ""
            if has_depth and "abs_rel" in metrics:
                depth_str = (f"  abs_rel={metrics['abs_rel']:.4f} "
                             f"a1={metrics['a1']:.4f}")
            print(f"[{i+1:>4}/{len(pairs)}] {pred_path.name} | "
                  f"PSNR={metrics['psnr']:.2f}  "
                  f"SSIM={metrics['ssim']:.4f}  "
                  f"LPIPS={metrics['lpips']:.4f}"
                  f"{depth_str}")

        except Exception as e:
            print(f"  ERROR on {pred_path.name}: {e} — skipping.")

    return all_results


# ──────────────────────────────────────────────
# Aggregation + reporting
# ──────────────────────────────────────────────

def aggregate(results: list) -> dict:
    keys = [k for k in results[0].keys() if k != "frame"]
    return {k: np.mean([r[k] for r in results if k in r]) for k in keys}


def print_summary(agg: dict):
    print("\n" + "=" * 55)
    print("  SUMMARY (mean over all frames)")
    print("=" * 55)
    print(f"  PSNR      : {agg['psnr']:.4f}  dB       ↑ higher is better")
    print(f"  SSIM      : {agg['ssim']:.4f}           ↑ higher is better")
    print(f"  LPIPS     : {agg['lpips']:.4f}           ↓ lower  is better")
    if "abs_rel" in agg:
        print(f"  Abs Rel   : {agg['abs_rel']:.4f}           ↓ lower  is better")
        print(f"  Sq  Rel   : {agg['sq_rel']:.4f}           ↓ lower  is better")
        print(f"  RMSE      : {agg['rmse']:.4f}           ↓ lower  is better")
        print(f"  RMSE log  : {agg['rmse_log']:.4f}           ↓ lower  is better")
        print(f"  δ < 1.25  : {agg['a1']:.4f}           ↑ higher is better")
        print(f"  δ < 1.25² : {agg['a2']:.4f}           ↑ higher is better")
        print(f"  δ < 1.25³ : {agg['a3']:.4f}           ↑ higher is better")
    print("=" * 55)


def save_csv(results: list, agg: dict, out_path: str):
    import csv
    keys = [k for k in results[0].keys()]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        # Append mean row
        mean_row = {"frame": "MEAN", **{k: f"{v:.6f}" for k, v in agg.items()}}
        writer.writerow(mean_row)
    print(f"Saved metrics to {out_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate NeRF/GS renderings: PSNR, SSIM, LPIPS, depth metrics.\n"
            "Depth can be computed on-the-fly with VGGT (--vggt-depth) or loaded\n"
            "from pre-existing files (--pred-depth-dir, removed in favour of VGGT)."
        )
    )
    parser.add_argument("--pred-dir",      required=True,
                        help="Directory of predicted RGB frames.")
    parser.add_argument("--gt-dir",        required=True,
                        help="Directory of ground-truth RGB frames.")
    parser.add_argument("--gt-depth-dir",  default=None,
                        help="Directory of ground-truth depth maps (required for depth metrics).")
    parser.add_argument("--vggt-depth",    action="store_true",
                        help="Use VGGT to generate depth maps for predicted frames "
                             "instead of loading them from disk.")
    parser.add_argument("--depth-ext",     default=".png",
                        help="Extension for depth map files (default: .png).")
    parser.add_argument("--min-depth",     type=float, default=1e-3)
    parser.add_argument("--max-depth",     type=float, default=50.0)
    parser.add_argument("--device",        default="cuda",
                        help="Torch device (cuda / cpu).")
    parser.add_argument("--out-csv",       default=None,
                        help="Optional path to save per-frame + mean metrics as CSV.")
    args = parser.parse_args()

    results = evaluate_directory(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        gt_depth_dir=args.gt_depth_dir,
        use_vggt_depth=args.vggt_depth,
        device=args.device,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        depth_ext=args.depth_ext,
    )

    if not results:
        print("No results — check your input directories.")
        return

    agg = aggregate(results)
    print_summary(agg)

    if args.out_csv:
        save_csv(results, agg, args.out_csv)


if __name__ == "__main__":
    main()