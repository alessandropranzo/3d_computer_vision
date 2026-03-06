#!/usr/bin/env python3
"""
Visualize 3D Gaussian Splatting scene from CF-3DGS output.

Supports two input formats:
  1. PLY file  -- output/progressive/.../point_cloud/iteration_XXXXX/point_cloud.ply
  2. Checkpoint -- output/progressive/.../chkpnt/ep00_init.pth

Can also overlay the camera trajectory from the pose file.

Usage:
    python visualize_3dgs.py --ply output/progressive/my_room/point_cloud.ply
    python visualize_3dgs.py --checkpoint output/progressive/my_room/chkpnt/ep00_init.pth
    python visualize_3dgs.py --checkpoint ... --pose output/.../pose/ep00_init.pth
    python visualize_3dgs.py --ply ... --export scene_export.ply
"""

import argparse
import os
import sys

import numpy as np


# SH DC coefficient (zeroth order spherical harmonic)
C0 = 0.28209479177387814


def sh_dc_to_rgb(sh_dc):
    """Convert SH DC coefficients to RGB colors in [0, 1]."""
    return np.clip(sh_dc * C0 + 0.5, 0.0, 1.0)


def load_from_ply(ply_path):
    """Load Gaussian positions and colors from a 3DGS PLY file."""
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

    try:
        f_dc_0 = vertex["f_dc_0"]
        f_dc_1 = vertex["f_dc_1"]
        f_dc_2 = vertex["f_dc_2"]
        sh_dc = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)
        colors = sh_dc_to_rgb(sh_dc)
    except ValueError:
        colors = np.ones_like(xyz) * 0.5

    opacities = None
    try:
        opacities = 1.0 / (1.0 + np.exp(-vertex["opacity"]))
    except ValueError:
        pass

    return xyz, colors, opacities


def load_from_checkpoint(ckpt_path):
    """Load Gaussian positions and colors from a CF-3DGS checkpoint.

    The checkpoint is a tuple from GaussianModel.capture():
        (active_sh_degree, _xyz, _features_dc, _features_rest,
         _scaling, _rotation, _opacity, max_radii2D,
         xyz_gradient_accum, denom, optimizer_state, spatial_lr_scale)
    """
    import torch

    data = torch.load(ckpt_path, map_location="cpu")

    if isinstance(data, tuple) and len(data) >= 7:
        _xyz = data[1]
        _features_dc = data[2]
        _opacity = data[6]
    else:
        print("Error: unrecognized checkpoint format.")
        sys.exit(1)

    xyz = _xyz.numpy()

    # _features_dc shape: (N, 1, 3) or (N, 3, 1) depending on version
    sh_dc = _features_dc.squeeze().numpy()
    if sh_dc.ndim == 2 and sh_dc.shape[1] == 3:
        colors = sh_dc_to_rgb(sh_dc)
    elif sh_dc.ndim == 2 and sh_dc.shape[0] == 3:
        colors = sh_dc_to_rgb(sh_dc.T)
    else:
        colors = np.ones_like(xyz) * 0.5

    opacities = 1.0 / (1.0 + np.exp(-_opacity.numpy().flatten()))

    return xyz, colors, opacities


def load_poses(pose_path):
    """Load camera poses from pose checkpoint."""
    import torch

    data = torch.load(pose_path, map_location="cpu")
    w2c = data["poses_pred"]
    c2w = torch.inverse(w2c)
    positions = c2w[:, :3, 3].numpy()
    return positions


def filter_by_opacity(xyz, colors, opacities, threshold=0.1):
    """Remove Gaussians with low opacity."""
    if opacities is None:
        return xyz, colors
    mask = opacities > threshold
    return xyz[mask], colors[mask]


def visualize_open3d(xyz, colors, cam_positions=None):
    """Interactive Open3D visualization."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed. Install with: pip install open3d")
        sys.exit(1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    if cam_positions is not None:
        n = len(cam_positions)
        lines = [[i, i + 1] for i in range(n - 1)]
        line_colors = [[1.0, 0.0, 0.0]] * len(lines)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(cam_positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        geometries.append(line_set)

        traj_extent = np.linalg.norm(
            cam_positions.max(0) - cam_positions.min(0))
        for i in [0, n - 1]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=traj_extent * 0.01)
            sphere.translate(cam_positions[i])
            color = [0.0, 1.0, 0.0] if i == 0 else [1.0, 0.0, 0.0]
            sphere.paint_uniform_color(color)
            geometries.append(sphere)

    scene_extent = np.linalg.norm(xyz.max(0) - xyz.min(0))
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scene_extent * 0.05)
    geometries.append(origin)

    print(f"Displaying {len(xyz):,} Gaussians")
    print("Open3D viewer controls:")
    print("  Left-click + drag  : Rotate")
    print("  Scroll             : Zoom")
    print("  Shift + left-click : Pan")
    print("  Q                  : Quit")
    o3d.visualization.draw_geometries(geometries,
                                      window_name="3D Gaussian Splatting Scene")


def export_colored_ply(xyz, colors, output_path):
    """Export a simple colored PLY viewable in MeshLab, CloudCompare, etc."""
    from plyfile import PlyData, PlyElement

    rgb_uint8 = (colors * 255).astype(np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(len(xyz), dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["red"] = rgb_uint8[:, 0]
    elements["green"] = rgb_uint8[:, 1]
    elements["blue"] = rgb_uint8[:, 2]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)
    print(f"Exported colored PLY to '{output_path}' ({len(xyz):,} points)")
    print("  Open in: MeshLab, CloudCompare, or upload to "
          "https://playcanvas.com/supersplat/editor")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CF-3DGS 3D Gaussian scene")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ply", help="Path to a 3DGS PLY file")
    group.add_argument("--checkpoint", "--ckpt",
                       help="Path to a CF-3DGS model checkpoint (.pth)")

    parser.add_argument("--pose", default=None,
                        help="Path to pose checkpoint to overlay camera "
                             "trajectory")
    parser.add_argument("--opacity-threshold", type=float, default=0.05,
                        help="Remove Gaussians with opacity below this "
                             "threshold (default: 0.05)")
    parser.add_argument("--max-points", type=int, default=500000,
                        help="Subsample to this many points for display "
                             "(default: 500000)")
    parser.add_argument("--export", default=None,
                        help="Export a simplified colored PLY to this path")
    args = parser.parse_args()

    if args.ply:
        print(f"Loading PLY from '{args.ply}'...")
        xyz, colors, opacities = load_from_ply(args.ply)
    else:
        print(f"Loading checkpoint from '{args.checkpoint}'...")
        xyz, colors, opacities = load_from_checkpoint(args.checkpoint)

    print(f"  Total Gaussians: {len(xyz):,}")

    xyz, colors = filter_by_opacity(xyz, colors, opacities,
                                    args.opacity_threshold)
    print(f"  After opacity filter (>{args.opacity_threshold}): "
          f"{len(xyz):,}")

    if len(xyz) > args.max_points:
        indices = np.random.choice(len(xyz), args.max_points, replace=False)
        indices.sort()
        xyz = xyz[indices]
        colors = colors[indices]
        print(f"  Subsampled to: {len(xyz):,}")

    cam_positions = None
    if args.pose:
        print(f"Loading camera poses from '{args.pose}'...")
        cam_positions = load_poses(args.pose)
        print(f"  Camera positions: {len(cam_positions)}")

    if args.export:
        export_colored_ply(xyz, colors, args.export)

    visualize_open3d(xyz, colors, cam_positions)


if __name__ == "__main__":
    main()
