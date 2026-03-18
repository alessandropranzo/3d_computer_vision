#!/usr/bin/env python3
"""
Visualize the estimated camera trajectory from CF-3DGS training output.

The pose file (e.g. output/progressive/.../pose/ep00_init.pth) contains:
  - poses_pred: (N, 4, 4) tensor of estimated world-to-camera matrices
  - poses_gt:   (N, 4, 4) tensor of ground-truth poses (empty for custom data)

Usage:
    python visualize_trajectory.py --pose output/progressive/my_room/pose/ep00_init.pth
    python visualize_trajectory.py --pose output/progressive/my_room/pose/ep00_init.pth --interactive
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_poses(pose_path):
    """Load poses from a CF-3DGS pose checkpoint."""
    data = torch.load(pose_path, map_location="cpu")

    w2c = data["poses_pred"]
    c2w = torch.inverse(w2c)
    positions = c2w[:, :3, 3].numpy()
    rotations = c2w[:, :3, :3].numpy()

    has_gt = ("poses_gt" in data and data["poses_gt"] is not None
              and len(data["poses_gt"]) > 0)
    gt_positions = None
    if has_gt:
        try:
            gt_c2w = torch.inverse(data["poses_gt"])
            gt_positions = gt_c2w[:, :3, 3].numpy()
        except Exception:
            pass

    return positions, rotations, gt_positions


def draw_camera_frustum(ax, position, rotation, size=0.05, color="blue"):
    """Draw a small camera frustum at the given pose."""
    hw = size * 0.5
    hh = size * 0.35
    d = size

    corners_local = np.array([
        [-hw, -hh, d],
        [hw, -hh, d],
        [hw, hh, d],
        [-hw, hh, d],
    ])

    corners_world = (rotation @ corners_local.T).T + position

    for corner in corners_world:
        ax.plot3D(*zip(position, corner), color=color, linewidth=0.5,
                  alpha=0.4)

    verts = [corners_world.tolist()]
    poly = Poly3DCollection(verts, alpha=0.1, facecolor=color,
                            edgecolor=color, linewidth=0.5)
    ax.add_collection3d(poly)


def plot_trajectory_matplotlib(positions, rotations, gt_positions=None,
                               save_path=None, show_frustums=True):
    """Plot camera trajectory in 3D using matplotlib."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    n = len(positions)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            "-", color="royalblue", linewidth=2, label="Estimated trajectory",
            zorder=5)

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=np.arange(n), cmap="viridis", s=15, zorder=10)

    ax.scatter(*positions[0], color="green", s=100, marker="^",
               label="Start", zorder=15)
    ax.scatter(*positions[-1], color="red", s=100, marker="v",
               label="End", zorder=15)

    if show_frustums:
        step = max(1, n // 20)
        frustum_size = np.linalg.norm(positions.max(0) - positions.min(0)) * 0.03
        for i in range(0, n, step):
            draw_camera_frustum(ax, positions[i], rotations[i],
                                size=frustum_size, color=colors[i])

    if gt_positions is not None:
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2],
                "--", color="red", linewidth=2, label="Ground truth",
                alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Estimated Camera Trajectory")
    ax.legend(loc="upper left")

    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(0, n - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Frame index")

    set_equal_aspect(ax, positions)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory plot to '{save_path}'")
    plt.show()


def set_equal_aspect(ax, points):
    """Set equal aspect ratio for 3D axes."""
    center = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() * 0.6
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)


def visualize_open3d(positions, rotations):
    """Interactive visualization using Open3D."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed. Install with: pip install open3d")
        sys.exit(1)

    n = len(positions)
    geometries = []

    points = positions.tolist()
    lines = [[i, i + 1] for i in range(n - 1)]
    colors_line = plt.cm.viridis(np.linspace(0, 1, len(lines)))[:, :3].tolist()

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors_line)
    geometries.append(line_set)

    traj_scale = np.linalg.norm(positions.max(0) - positions.min(0))
    cam_size = traj_scale * 0.02

    step = max(1, n // 30)
    for i in range(0, n, step):
        frustum = create_camera_frustum_o3d(positions[i], rotations[i],
                                            cam_size)
        t = i / max(1, n - 1)
        color = plt.cm.viridis(t)[:3]
        frustum.paint_uniform_color(color)
        geometries.append(frustum)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=traj_scale * 0.1)
    geometries.append(origin)

    print("Open3D viewer controls:")
    print("  Left-click + drag  : Rotate")
    print("  Scroll             : Zoom")
    print("  Shift + left-click : Pan")
    print("  Q                  : Quit")
    o3d.visualization.draw_geometries(geometries,
                                      window_name="Camera Trajectory")


def create_camera_frustum_o3d(position, rotation, size=0.05):
    """Create a wireframe camera frustum for Open3D."""
    import open3d as o3d

    hw = size * 0.5
    hh = size * 0.35
    d = size

    points = np.array([
        [0, 0, 0],
        [-hw, -hh, d],
        [hw, -hh, d],
        [hw, hh, d],
        [-hw, hh, d],
    ])

    points_world = (rotation @ points.T).T + position

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CF-3DGS estimated camera trajectory")
    parser.add_argument("--pose", "-p", required=True,
                        help="Path to the pose checkpoint "
                             "(e.g. output/.../pose/ep00_init.pth)")
    parser.add_argument("--save", "-s", default=None,
                        help="Save the matplotlib plot to this path "
                             "(e.g. trajectory.png)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Open an interactive Open3D viewer")
    parser.add_argument("--no-frustums", action="store_true",
                        help="Disable camera frustum rendering")
    args = parser.parse_args()

    print(f"Loading poses from '{args.pose}'...")
    positions, rotations, gt_positions = load_poses(args.pose)
    print(f"  Found {len(positions)} camera poses")

    if gt_positions is not None:
        print(f"  Found {len(gt_positions)} ground-truth poses")

    if args.interactive:
        visualize_open3d(positions, rotations)
    else:
        save_path = args.save or "trajectory.png"
        plot_trajectory_matplotlib(positions, rotations, gt_positions,
                                   save_path=save_path,
                                   show_frustums=not args.no_frustums)


if __name__ == "__main__":
    main()
