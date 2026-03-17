import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import argparse


def generate_predictions(images, device="cuda", dtype=torch.float32):
    print(f"\nLoading model on {device}")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    S, C, H, W = images.shape

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(50, H-50, steps=10),
        torch.linspace(50, W-50, steps=10),
        indexing='ij'
    )
    query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim = -1).to(device)

    print("\nRunning inference")
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.no_grad():
            predictions = model(images, query_points=query_points)

    return predictions


def save_predictions(pcd, cam_trajectory, traj_1, traj_2, out_dir):

    o3d.io.write_point_cloud(os.path.join(out_dir, "scene_representation.ply"), pcd)
    print(f"\nSaved scene_representation.ply")
    
    np.save(os.path.join(out_dir, "camera_trajectory_v1.npy"), traj_1)
    np.save(os.path.join(out_dir, "camera_trajectory_v2.npy"), traj_2)
    print(f"Saved camera trajectories to: {os.path.join(out_dir, 'camera_trajectory_v1.npy')} and {os.path.join(out_dir, 'camera_trajectory_v2.npy')}")


def main():
    parser = argparse.ArgumentParser(description="Extract Visualizations from VGGT")
    parser.add_argument("--path1", type=str, default="./data/images/room1/", help="Path to first image folder")
    parser.add_argument("--path2", type=str, default=None, help="Path to second image folder")
    parser.add_argument("--alternate", action="store_true", help="Alternate frames from path1 and path2 instead of concatenating")
    parser.add_argument("--out_dir", type=str, default="predictions_visuals")
    args = parser.parse_args()

    image_names1 = [os.path.join(args.path1, f) for f in sorted(os.listdir(args.path1)) if os.path.isfile(os.path.join(args.path1, f))]

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    print(f"Loading {len(image_names1)} images from path1...")
    images1 = load_and_preprocess_images(image_names1).to(device)
    
    if args.path2 is not None:
        image_names2 = [os.path.join(args.path2, f) for f in sorted(os.listdir(args.path2)) if os.path.isfile(os.path.join(args.path2, f))]
        print(f"Loading {len(image_names2)} images from path2...")
        images2 = load_and_preprocess_images(image_names2).to(device)
        
        if args.alternate:
            s1, c1, h1, w1 = images1.shape
            s2, c2, h2, w2 = images2.shape
            min_s = min(s1, s2)
            
            # Interleave frames: [img1[0], img2[0], img1[1], img2[1], ...] up to min length
            images = torch.empty((2 * min_s, c1, h1, w1), device=device, dtype=images1.dtype)
            images[0::2] = images1[:min_s]
            images[1::2] = images2[:min_s]
        else:
            images = torch.cat([images1, images2], dim=0)
    else:
        images = images1

    predictions = generate_predictions(images, device, dtype)
            
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if "world_points" in predictions:

        wp = predictions["world_points"].cpu().float()
        colors = images.cpu().permute(0, 2, 3, 1)
        S, C, H, W = images.shape

        if args.path2:
            if args.alternate:
                wp1 = wp[0::2]
                wp2 = wp[1::2]
            else:
                wp1 = wp[:len(image_names1)]
                wp2 = wp[len(image_names1):]

        if len(wp.shape) == 5:
            wp = wp.squeeze(0)

        all_points = []
        all_colors = []

        for frame_idx in range(S):
            points_3d = wp[frame_idx].reshape(-1, 3).numpy()
            point_colors = colors[frame_idx].reshape(-1, 3).numpy()

            all_points.append(points_3d)
            all_colors.append(point_colors)

        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    else:
        raise Exception("No world points in predictions")

    if "pose_enc" in predictions:
        pose_enc = predictions["pose_enc"].cpu()
        if len(pose_enc.shape) == 2:
            pose_enc = pose_enc.unsqueeze(0) # (1, S, 9)

        extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W), build_intrinsics=False)
        R = extrinsics[0, :, :3, :3] # (B, S, 3, 3)
        T = extrinsics[0, :, :3, 3] # (B, S, 3)

        # Invert to get camera-to-world
        R_inv = R.transpose(1, 2)
        t_inv = -torch.bmm(R_inv, T.unsqueeze(-1)).squeeze(-1)

        cam_trajectory = t_inv.numpy()

        if args.path2 is not None:
            s1_len = images1.shape[0]
            if args.alternate:
                traj_1 = cam_trajectory[0::2]
                traj_2 = cam_trajectory[1::2]
            else:
                traj_1 = cam_trajectory[:s1_len]
                traj_2 = cam_trajectory[s1_len:]

    else:
        raise Exception("No pose_enc in predictions")

    save_predictions(pcd, cam_trajectory, traj_1, traj_2, out_dir)


if __name__ == "__main__":
    main()
