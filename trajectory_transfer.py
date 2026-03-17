#!/usr/bin/env python3
"""
Trajectory Transfer
===================
Compute novel camera poses by transferring the spatial relationship between
two rooms (observed jointly by VGGT) into a standalone coordinate system.

Stages:
  1. Load camera parameters from VGGT predictions (merged and room1-only).
  2. Match cameras between rooms in merged space (Hungarian algorithm).
  3. Compute trajectory difference (position + rotation deltas).
  4. Procrustes alignment between merged and standalone coordinate systems.
  5. Apply aligned deltas to standalone poses → novel camera poses.

Outputs (written to --output_dir):
  novel_extrinsics.npy, novel_intrinsics.npy,
  match_row_ind.npy, match_col_ind.npy,
  alignment.npz  (scale, rotation, translation)

Usage:
    python trajectory_transfer.py \\
        --merged_dir predictions_visuals/merged \\
        --room1_dir predictions_visuals/room1 \\
        --images_room1 data/images/room1 \\
        --output_dir transfer_results
"""

import os
import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# Data Loading
# ============================================================

def load_predictions(pred_dir):
    """Load all .npy camera data from a VGGT predictions directory."""
    data = {}
    for name in [
        "extrinsics", "intrinsics", "camera_trajectory_merged",
        "extrinsics_v1", "extrinsics_v2", "intrinsics_v1", "intrinsics_v2",
        "camera_trajectory_v1", "camera_trajectory_v2",
    ]:
        path = os.path.join(pred_dir, f"{name}.npy")
        if os.path.exists(path):
            data[name] = np.load(path)
    return data


def count_images(image_dir):
    """Count image files in a directory."""
    return len([
        f for f in sorted(os.listdir(image_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def extrinsics_to_centers(extrinsics):
    """(N, 3, 4) extrinsics [R|t] -> (N, 3) camera centres.  C = -R^T t."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)


# ============================================================
# Camera Matching & Trajectory Difference
# ============================================================

def match_cameras_hungarian(centers1, centers2):
    """Optimal one-to-one matching by Euclidean distance (Hungarian).

    Returns (row_ind, col_ind) arrays of matched indices.
    """
    dist = np.linalg.norm(centers1[:, None] - centers2[None, :], axis=-1)
    return linear_sum_assignment(dist)


def compute_trajectory_deltas(extrinsics1, extrinsics2, row_ind, col_ind):
    """Position and rotation deltas between matched pairs.

    Returns:
        delta_pos  (M, 3)   – C2[col] - C1[row]
        delta_rot  (M, 3, 3) – R2 @ R1^T  (so that R2 = delta_R @ R1)
    """
    c1 = extrinsics_to_centers(extrinsics1[row_ind])
    c2 = extrinsics_to_centers(extrinsics2[col_ind])
    delta_pos = c2 - c1

    R1 = extrinsics1[row_ind, :3, :3]
    R2 = extrinsics2[col_ind, :3, :3]
    delta_rot = np.einsum("nij,nkj->nik", R2, R1)  # R2 @ R1^T
    return delta_pos, delta_rot


# ============================================================
# Procrustes Alignment
# ============================================================

def procrustes_alignment(source, target):
    """SVD-based Procrustes: target ≈ s * R @ source + t.

    Returns (scale, R_3x3, t_3).
    """
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src_c = source - mu_s
    tgt_c = target - mu_t

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, d])
    R = Vt.T @ S @ U.T

    scale = np.trace(R @ H) / np.trace(src_c.T @ src_c)
    t = mu_t - scale * R @ mu_s
    return scale, R, t


def apply_alignment_to_deltas(deltas, scale, rotation):
    """Rotate and scale delta vectors into the target coordinate system."""
    return scale * (rotation @ deltas.T).T


# ============================================================
# Novel Pose Computation
# ============================================================

def compute_novel_poses(extri_room1, intri_room1, row_ind, delta_aligned):
    """Shift standalone camera centres by aligned deltas.

    Returns (novel_extrinsics, novel_intrinsics).
    """
    novel_extri = extri_room1[row_ind].copy()
    novel_intri = intri_room1[row_ind].copy()

    R_s = novel_extri[:, :3, :3]
    t_s = novel_extri[:, :3, 3]
    c_orig = -np.einsum("nij,nj->ni", R_s.transpose(0, 2, 1), t_s)
    c_novel = c_orig + delta_aligned
    t_novel = -np.einsum("nij,nj->ni", R_s, c_novel)
    novel_extri[:, :3, 3] = t_novel

    return novel_extri, novel_intri


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trajectory Transfer – compute novel camera poses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--merged_dir", default="predictions_visuals/merged")
    parser.add_argument("--room1_dir", default="predictions_visuals/room1")
    parser.add_argument("--images_room1", default="data/images/room1")
    parser.add_argument("--output_dir", default="transfer_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Stage 1: Load predictions ──────────────────────────────────────
    print("=" * 60)
    print("Stage 1 – Loading predictions")
    print("=" * 60)

    merged = load_predictions(args.merged_dir)
    room1 = load_predictions(args.room1_dir)

    extri_merged = merged["extrinsics"]
    extri_room1 = room1["extrinsics"]
    intri_room1 = room1["intrinsics"]

    n1 = count_images(args.images_room1)
    n_total = extri_merged.shape[0]
    n2 = n_total - n1

    extri_m_r1 = extri_merged[:n1]
    extri_m_r2 = extri_merged[n1:]
    centers_m_r1 = extrinsics_to_centers(extri_m_r1)
    centers_m_r2 = extrinsics_to_centers(extri_m_r2)
    centers_standalone = extrinsics_to_centers(extri_room1)

    print(f"  Merged frames : {n_total}  (room1={n1}, room2={n2})")
    print(f"  Standalone r1 : {extri_room1.shape[0]} frames")

    # ── Stage 2: Camera matching ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 2 – Hungarian camera matching")
    print("=" * 60)

    row_ind, col_ind = match_cameras_hungarian(centers_m_r1, centers_m_r2)
    match_dists = np.linalg.norm(
        centers_m_r1[row_ind] - centers_m_r2[col_ind], axis=-1
    )
    print(f"  Matched {len(row_ind)} pairs")
    print(f"  Distances: mean={match_dists.mean():.4f}  "
          f"min={match_dists.min():.4f}  max={match_dists.max():.4f}")

    np.save(os.path.join(args.output_dir, "match_row_ind.npy"), row_ind)
    np.save(os.path.join(args.output_dir, "match_col_ind.npy"), col_ind)

    # ── Stage 3: Trajectory difference ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 3 – Trajectory deltas")
    print("=" * 60)

    delta_pos, delta_rot = compute_trajectory_deltas(
        extri_m_r1, extri_m_r2, row_ind, col_ind
    )
    print(f"  Pos-delta mean norm : {np.linalg.norm(delta_pos, axis=-1).mean():.4f}")

    # ── Stage 4: Procrustes alignment ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 4 – Procrustes alignment (merged → standalone)")
    print("=" * 60)

    scale, R_align, t_align = procrustes_alignment(centers_m_r1, centers_standalone)
    aligned = scale * (R_align @ centers_m_r1.T).T + t_align
    residual = np.linalg.norm(aligned - centers_standalone, axis=-1).mean()
    print(f"  Scale    : {scale:.6f}")
    print(f"  Residual : {residual:.6f}")

    delta_aligned = apply_alignment_to_deltas(delta_pos, scale, R_align)
    print(f"  Aligned-delta mean norm : {np.linalg.norm(delta_aligned, axis=-1).mean():.4f}")

    np.savez(
        os.path.join(args.output_dir, "alignment.npz"),
        scale=scale, rotation=R_align, translation=t_align,
    )

    # ── Stage 5: Novel camera poses ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 5 – Computing novel camera poses")
    print("=" * 60)

    novel_extri, novel_intri = compute_novel_poses(
        extri_room1, intri_room1, row_ind, delta_aligned,
    )

    np.save(os.path.join(args.output_dir, "novel_extrinsics.npy"), novel_extri)
    np.save(os.path.join(args.output_dir, "novel_intrinsics.npy"), novel_intri)
    print(f"  {novel_extri.shape[0]} novel poses saved")
    print(f"  Mean position shift: {np.linalg.norm(delta_aligned, axis=-1).mean():.4f}")

    print(f"\nDone!  Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
