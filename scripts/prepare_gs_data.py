#!/usr/bin/env python3
"""
Prepare COLMAP Dataset for 3D Gaussian Splatting
================================================
This script converts the VGGT predictions (extrinsics, intrinsics, images)
and the 3D point cloud (`scene_representation.ply`) into a standard COLMAP
`sparse/0` format. This allows you to train a standard 3D Gaussian Splatting
model (e.g. INRIA's gaussian-splatting) on the room 1 data.

Usage:
    python prepare_gs_data.py \\
        --room1_dir predictions_visuals/room1 \\
        --images_dir data/images/room1 \\
        --output_dir data/colmap_room1

After running this, you can train 3DGS using the official repo:
    python train.py -s ../data/colmap_room1 -m output/gs_room1
"""

import os
import shutil
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Convert VGGT to COLMAP format")
    parser.add_argument("--room1_dir", default="results/vggt_predictions/room1")
    parser.add_argument("--images_dir", default="data/images/room1")
    parser.add_argument("--output_dir", default="data/colmap_room1")
    args = parser.parse_args()

    sparse_dir = os.path.join(args.output_dir, "sparse", "0")
    images_out_dir = os.path.join(args.output_dir, "images")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_out_dir, exist_ok=True)

    # 1. Load predictions
    extri = np.load(os.path.join(args.room1_dir, "extrinsics.npy"))  # (N, 3, 4)
    intri = np.load(os.path.join(args.room1_dir, "intrinsics.npy"))  # (N, 3, 3)

    # 2. Get images
    fnames = sorted([f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if len(fnames) != extri.shape[0]:
        print(f"Warning: Number of images ({len(fnames)}) != number of poses ({extri.shape[0]})")
    
    # Read first image to get dimensions
    first_img = Image.open(os.path.join(args.images_dir, fnames[0]))
    W, H = first_img.size

    # In VGGT training scripts, we resize images to W=518, H divisible by 14.
    # To be consistent with how VGGT evaluated the extrinsics/intrinsics, we should copy and resize the images.
    target_width = 518
    target_height = round(H * (target_width / W) / 14) * 14
    if target_height > target_width:
        start_y = (target_height - target_width) // 2
        crop_box = (0, start_y, target_width, start_y + target_width)
        W_out, H_out = target_width, target_width
    else:
        crop_box = None
        W_out, H_out = target_width, target_height

    print(f"Processing {len(fnames)} images. Output dimensions: {W_out}x{H_out}")

    # 3. Write cameras.txt
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(len(fnames)):
            cam_id = i + 1
            K = intri[i]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            f.write(f"{cam_id} PINHOLE {W_out} {H_out} {fx} {fy} {cx} {cy}\n")

    # 4. Write images.txt
    images_txt = os.path.join(sparse_dir, "images.txt")
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, fname in enumerate(fnames):
            img_id = i + 1
            cam_id = i + 1
            
            # Save resized image to output
            img = Image.open(os.path.join(args.images_dir, fname)).convert("RGB")
            img = img.resize((target_width, target_height), Image.BICUBIC)
            if crop_box is not None:
                img = img.crop(crop_box)
            img.save(os.path.join(images_out_dir, fname))

            R = extri[i, :3, :3]
            t = extri[i, :3, 3]
            
            # Convert R to quaternion (qx, qy, qz, qw) -> (qw, qx, qy, qz)
            rot = Rotation.from_matrix(R)
            qx, qy, qz, qw = rot.as_quat()
            
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {fname}\n")
            f.write("\n")  # Empty points2D line

    # 5. Copy and fix point cloud (add normals)
    ply_src = os.path.join(args.room1_dir, "scene_representation.ply")
    ply_dst = os.path.join(sparse_dir, "points3D.ply")
    if os.path.exists(ply_src):
        # We need to read it with Open3D, then manually write a PLY that has normals
        # so that gaussian-splatting's fetchPly doesn't crash.
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(ply_src)
            pcd.estimate_normals()
            
            # gaussian-splatting expects specific types (float32 for float, uint8 for colors)
            from plyfile import PlyData, PlyElement
            xyz = np.asarray(pcd.points).astype(np.float32)
            normals = np.asarray(pcd.normals).astype(np.float32)
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                     ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            
            elements = np.empty(len(xyz), dtype=dtype)
            elements['x'] = xyz[:, 0]
            elements['y'] = xyz[:, 1]
            elements['z'] = xyz[:, 2]
            elements['nx'] = normals[:, 0]
            elements['ny'] = normals[:, 1]
            elements['nz'] = normals[:, 2]
            elements['red'] = colors[:, 0]
            elements['green'] = colors[:, 1]
            elements['blue'] = colors[:, 2]
            
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el], text=False).write(ply_dst)
            print(f"Fixed and saved {ply_src} to {ply_dst} with normals")
        except ImportError:
            print("open3d or plyfile not found. Falling back to random init.")
            with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
                f.write("# 3D point list with one line of data per point:\n")
                f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    else:
        print(f"Warning: {ply_src} not found. 3DGS will initialize from random points.")
        # Create empty points3D.txt as fallback
        with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

    print(f"\nDone! COLMAP dataset created at: {args.output_dir}")
    print("You can now train 3DGS using the official repo:")
    print(f"  python train.py -s $(pwd)/{args.output_dir} -m $(pwd)/output/gs_room1")

if __name__ == "__main__":
    main()
