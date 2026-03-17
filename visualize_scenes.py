#!/usr/bin/env python3
"""
3D Scene & Trajectory Visualizer
=================================
Interactive Plotly visualizations of VGGT point clouds and camera trajectories.

Generates standalone HTML files you can open in a browser while the NeRF trains.

Visualizations produced:
  1. Room 1 point cloud + standalone camera trajectory
  2. Merged point cloud + room 1 & room 2 trajectories
  3. Room 1 point cloud + standalone trajectory + transferred trajectory

Usage:
    python visualize_scenes.py

    python visualize_scenes.py \
        --merged_dir predictions_visuals/merged \
        --room1_dir predictions_visuals/room1 \
        --images_room1 data/images/room1 \
        --nerf_results_dir nerf_results \
        --output_dir visualizations
"""

import os
import argparse

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment


# ── Geometry helpers (same as trajectory_transfer.py) ───────────────────

def extrinsics_to_centers(extrinsics):
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)


def match_cameras_hungarian(centers1, centers2):
    dist = np.linalg.norm(centers1[:, None] - centers2[None, :], axis=-1)
    return linear_sum_assignment(dist)


def compute_trajectory_deltas(extrinsics1, extrinsics2, row_ind, col_ind):
    c1 = extrinsics_to_centers(extrinsics1[row_ind])
    c2 = extrinsics_to_centers(extrinsics2[col_ind])
    return c2 - c1


def procrustes_alignment(source, target):
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


# ── Point-cloud loading ─────────────────────────────────────────────────

def load_point_cloud(ply_path, subsample=150_000):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    n = len(points)
    print(f"  Loaded {n:,} points from {ply_path}")
    if n > subsample:
        idx = np.random.choice(n, subsample, replace=False)
        points, colors = points[idx], colors[idx]
        print(f"  Subsampled to {subsample:,} for plotting")
    return points, colors


# ── Plotly trace builders ───────────────────────────────────────────────

def point_cloud_trace(points, colors, name="Point Cloud", size=1.2, opacity=0.85):
    rgb_strings = [
        f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in colors
    ]
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=size, color=rgb_strings, opacity=opacity),
        name=name,
        hoverinfo="skip",
    )


def trajectory_trace(centers, name, color, width=4, marker_size=5):
    return go.Scatter3d(
        x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
        mode="lines+markers",
        line=dict(color=color, width=width),
        marker=dict(size=marker_size, color=color, symbol="diamond"),
        name=name,
    )


def start_end_markers(centers, name_prefix, color):
    """Large sphere at start and cube at end of a trajectory."""
    traces = []
    traces.append(go.Scatter3d(
        x=[centers[0, 0]], y=[centers[0, 1]], z=[centers[0, 2]],
        mode="markers+text",
        marker=dict(size=10, color=color, symbol="circle"),
        text=[f"{name_prefix} start"],
        textposition="top center",
        showlegend=False,
        hoverinfo="text",
    ))
    traces.append(go.Scatter3d(
        x=[centers[-1, 0]], y=[centers[-1, 1]], z=[centers[-1, 2]],
        mode="markers+text",
        marker=dict(size=10, color=color, symbol="square"),
        text=[f"{name_prefix} end"],
        textposition="top center",
        showlegend=False,
        hoverinfo="text",
    ))
    return traces


def matching_lines_traces(centers_a, centers_b, row_ind, col_ind, color="gray", name="Matches"):
    """Dashed lines connecting matched camera pairs."""
    xs, ys, zs = [], [], []
    for r, c in zip(row_ind, col_ind):
        xs += [centers_a[r, 0], centers_b[c, 0], None]
        ys += [centers_a[r, 1], centers_b[c, 1], None]
        zs += [centers_a[r, 2], centers_b[c, 2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        name=name,
        opacity=0.4,
    )


# ── Figure assembly ─────────────────────────────────────────────────────

LAYOUT_DEFAULTS = dict(
    scene=dict(
        aspectmode="data",
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    template="plotly_dark",
)


def build_figure(traces, title=""):
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, **LAYOUT_DEFAULTS)
    return fig


def save_figure(fig, path):
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  Saved → {path}")


# ── Transferred trajectory computation ──────────────────────────────────

def compute_transferred_trajectory(merged_dir, room1_dir, images_room1_dir):
    """
    Re-run Stages 1-4 of the NeRF transfer pipeline (pure numpy, fast)
    to obtain the transferred camera centers in room1-standalone space.
    Returns (standalone_centers, transferred_centers, row_ind, col_ind).
    """
    extri_merged = np.load(os.path.join(merged_dir, "extrinsics.npy"))
    extri_room1 = np.load(os.path.join(room1_dir, "extrinsics.npy"))

    n1 = len([
        f for f in sorted(os.listdir(images_room1_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    extri_m_r1 = extri_merged[:n1]
    extri_m_r2 = extri_merged[n1:]

    centers_m_r1 = extrinsics_to_centers(extri_m_r1)
    centers_m_r2 = extrinsics_to_centers(extri_m_r2)
    centers_standalone = extrinsics_to_centers(extri_room1)

    row_ind, col_ind = match_cameras_hungarian(centers_m_r1, centers_m_r2)
    delta_pos = compute_trajectory_deltas(extri_m_r1, extri_m_r2, row_ind, col_ind)

    scale, R_align, _ = procrustes_alignment(centers_m_r1, centers_standalone)
    delta_aligned = scale * (R_align @ delta_pos.T).T

    c_orig = centers_standalone[row_ind]
    c_transferred = c_orig + delta_aligned

    return centers_standalone, c_transferred, row_ind, col_ind


def load_transferred_trajectory(nerf_results_dir):
    """Load pre-computed novel extrinsics from the NeRF pipeline output."""
    path = os.path.join(nerf_results_dir, "novel_extrinsics.npy")
    if not os.path.exists(path):
        return None
    novel_extri = np.load(path)
    return extrinsics_to_centers(novel_extri)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D scenes and camera trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--merged_dir", default="predictions_visuals/merged")
    parser.add_argument("--room1_dir", default="predictions_visuals/room1")
    parser.add_argument("--room2_dir", default="predictions_visuals/room2")
    parser.add_argument("--images_room1", default="data/images/room1")
    parser.add_argument("--transfer_dir", default="transfer_results",
                        help="Output of trajectory_transfer.py (novel poses)")
    parser.add_argument("--nerf_results_dir", default="nerf_results",
                        help="Output of nerf_render.py (fallback for novel poses)")
    parser.add_argument("--output_dir", default="visualizations")
    parser.add_argument("--subsample", type=int, default=150_000,
                        help="Max points to display per cloud")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    room1_ply = os.path.join(args.room1_dir, "scene_representation.ply")
    merged_ply = os.path.join(args.merged_dir, "scene_representation.ply")

    # ── Load extrinsics ────────────────────────────────────────────────
    extri_room1 = np.load(os.path.join(args.room1_dir, "extrinsics.npy"))
    centers_room1 = extrinsics_to_centers(extri_room1)

    extri_merged = np.load(os.path.join(args.merged_dir, "extrinsics.npy"))
    n1 = len([
        f for f in sorted(os.listdir(args.images_room1))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    centers_merged_r1 = extrinsics_to_centers(extri_merged[:n1])
    centers_merged_r2 = extrinsics_to_centers(extri_merged[n1:])

    # ── 1. Room 1 + standalone trajectory ──────────────────────────────
    print("=" * 60)
    print("Figure 1 – Room 1 scene + standalone camera trajectory")
    print("=" * 60)
    pts1, col1 = load_point_cloud(room1_ply, args.subsample)
    traces_1 = [
        point_cloud_trace(pts1, col1, name="Room 1"),
        trajectory_trace(centers_room1, "Standalone trajectory", "#00ccff"),
        *start_end_markers(centers_room1, "Standalone", "#00ccff"),
    ]
    fig1 = build_figure(traces_1, "Room 1 – Standalone Camera Trajectory")
    save_figure(fig1, os.path.join(args.output_dir, "room1_standalone.html"))

    # ── 2. Room 2 + standalone trajectory (if available) ────────────────
    room2_ply = os.path.join(args.room2_dir, "scene_representation.ply")
    room2_extri_path = os.path.join(args.room2_dir, "extrinsics.npy")
    if os.path.exists(room2_ply) and os.path.exists(room2_extri_path):
        print("\n" + "=" * 60)
        print("Figure 2 – Room 2 scene + standalone camera trajectory")
        print("=" * 60)
        extri_room2 = np.load(room2_extri_path)
        centers_room2 = extrinsics_to_centers(extri_room2)
        pts2, col2 = load_point_cloud(room2_ply, args.subsample)
        traces_2 = [
            point_cloud_trace(pts2, col2, name="Room 2"),
            trajectory_trace(centers_room2, "Standalone trajectory", "#ff6600"),
            *start_end_markers(centers_room2, "Standalone", "#ff6600"),
        ]
        fig2 = build_figure(traces_2, "Room 2 – Standalone Camera Trajectory")
        save_figure(fig2, os.path.join(args.output_dir, "room2_standalone.html"))
    else:
        print("\n  Skipping Room 2 standalone (no predictions found at "
              f"{args.room2_dir})")

    # ── 3. Merged scene + both room trajectories ───────────────────────
    print("\n" + "=" * 60)
    print("Figure 3 – Merged scene + room 1 & room 2 trajectories")
    print("=" * 60)
    pts_m, col_m = load_point_cloud(merged_ply, args.subsample)
    traces_3 = [
        point_cloud_trace(pts_m, col_m, name="Merged scene"),
        trajectory_trace(centers_merged_r1, "Room 1 (merged)", "#00ccff"),
        trajectory_trace(centers_merged_r2, "Room 2 (merged)", "#ff6600"),
        *start_end_markers(centers_merged_r1, "R1", "#00ccff"),
        *start_end_markers(centers_merged_r2, "R2", "#ff6600"),
    ]
    fig3 = build_figure(traces_3, "Merged Scene – Room 1 & Room 2 Trajectories")
    save_figure(fig3, os.path.join(args.output_dir, "merged_scene.html"))

    # ── 4. Room 1 + standalone + transferred trajectory ────────────────
    print("\n" + "=" * 60)
    print("Figure 4 – Room 1 scene + standalone vs. transferred trajectory")
    print("=" * 60)

    c_transferred = load_transferred_trajectory(args.transfer_dir)
    if c_transferred is None:
        c_transferred = load_transferred_trajectory(args.nerf_results_dir)

    row_ind = None
    if c_transferred is not None:
        print("  Loaded pre-computed novel extrinsics")
        for d in [args.transfer_dir, args.nerf_results_dir]:
            row_path = os.path.join(d, "match_row_ind.npy")
            if os.path.exists(row_path):
                row_ind = np.load(row_path)
                break
    else:
        print("  novel_extrinsics.npy not found – recomputing from scratch")
        centers_room1, c_transferred, row_ind, _ = compute_transferred_trajectory(
            args.merged_dir, args.room1_dir, args.images_room1,
        )

    traces_4 = [
        point_cloud_trace(pts1, col1, name="Room 1"),
        trajectory_trace(centers_room1, "Standalone trajectory", "#00ccff"),
        trajectory_trace(c_transferred, "Transferred trajectory", "#ff3366"),
        *start_end_markers(centers_room1, "Standalone", "#00ccff"),
        *start_end_markers(c_transferred, "Transferred", "#ff3366"),
    ]
    if row_ind is not None:
        traces_4.append(
            matching_lines_traces(
                centers_room1[row_ind],
                c_transferred,
                np.arange(len(c_transferred)),
                np.arange(len(c_transferred)),
                color="rgba(180,180,180,0.4)",
                name="Position shifts",
            )
        )
    fig4 = build_figure(traces_4, "Room 1 – Standalone vs. Transferred Trajectory")
    save_figure(fig4, os.path.join(args.output_dir, "room1_transferred.html"))

    print(f"\nDone! Open the HTML files in {args.output_dir}/ with your browser.")


if __name__ == "__main__":
    main()
