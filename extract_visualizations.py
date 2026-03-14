import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract Visualizations from VGGT")
    parser.add_argument("--path1", type=str, default="./data/images/room1/", help="Path to first image folder")
    parser.add_argument("--path2", type=str, default=None, help="Path to second image folder")
    parser.add_argument("--out_dir", type=str, default="predictions_visuals")
    args = parser.parse_args()

    image_names1 = [os.path.join(args.path1, f) for f in sorted(os.listdir(args.path1)) if os.path.isfile(os.path.join(args.path1, f))]

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
        
    print(f"Loading model on {device}...")
    # Initialize the model and load the pretrained weights.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    
    print(f"Loading {len(image_names1)} images from path1...")
    images1 = load_and_preprocess_images(image_names1).to(device)
    
    if args.path2 is not None:
        image_names2 = [os.path.join(args.path2, f) for f in sorted(os.listdir(args.path2)) if os.path.isfile(os.path.join(args.path2, f))]
        print(f"Loading {len(image_names2)} images from path2...")
        images2 = load_and_preprocess_images(image_names2).to(device)
        images = torch.cat([images1, images2], dim=0)
    else:
        images = images1
    
    S, C, H, W = images.shape
    # Create a grid of query points for tracking
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(50, H-50, steps=10), 
        torch.linspace(50, W-50, steps=10), 
        indexing='ij'
    )
    query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(device)
    
    print("Running inference...")
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.no_grad():
            predictions = model(images, query_points=query_points)
            
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

        
    # 2. Plot Depth Maps
    if "depth" in predictions:
        print("Plotting depth maps...")
        depth = predictions["depth"].cpu()
        # Ensure we have spatial dimensions (S, H, W)
        if len(depth.shape) == 5:    # (B, S, H, W, 1)
            depth = depth.squeeze(0).squeeze(-1)
        elif len(depth.shape) == 4:  # (S, H, W, 1) or (B, S, H, W)
            if depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            else:
                depth = depth.squeeze(0)
                
        depth_dir = os.path.join(out_dir, "depths")
        os.makedirs(depth_dir, exist_ok=True)
        
        S_d = depth.shape[0]
        frames_per_row = 4
        rows = (S_d + frames_per_row - 1) // frames_per_row
        
        fig, axes = plt.subplots(rows, frames_per_row, figsize=(4 * frames_per_row, 3 * rows))
        # Handle case where there's only 1 row or 1 column
        if rows == 1 and frames_per_row == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i in range(len(axes)):
            if i < S_d:
                im = axes[i].imshow(depth[i].numpy(), cmap='plasma')
                axes[i].set_title(f"Frame {i:02d}")
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            axes[i].axis('off')
        
        plt.tight_layout()
        depth_grid_path = os.path.join(depth_dir, "depth_grid.png")
        plt.savefig(depth_grid_path, dpi=150)
        plt.close()
        print(f"Depth grid saved to: {depth_grid_path}")
        
    # 3. Plot World Points Confidence (Optional extra)
    if "world_points_conf" in predictions:
        print("Plotting world points confidence...")
        wp_conf = predictions["world_points_conf"].cpu()
        if len(wp_conf.shape) == 4: # (B, S, H, W)
            wp_conf = wp_conf.squeeze(0)
            
        wp_dir = os.path.join(out_dir, "world_points_conf")
        os.makedirs(wp_dir, exist_ok=True)
        
        S_w = wp_conf.shape[0]
        rows = (S_w + frames_per_row - 1) // frames_per_row
        fig, axes = plt.subplots(rows, frames_per_row, figsize=(4 * frames_per_row, 3 * rows))
        if rows == 1 and frames_per_row == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i in range(len(axes)):
            if i < S_w:
                im = axes[i].imshow(wp_conf[i].numpy(), cmap='viridis', vmin=0, vmax=1)
                axes[i].set_title(f"Conf {i:02d}")
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            axes[i].axis('off')
            
        plt.tight_layout()
        wp_grid_path = os.path.join(wp_dir, "wp_conf_grid.png")
        plt.savefig(wp_grid_path, dpi=150)
        plt.close()
        print(f"World points confidence grid saved to: {wp_grid_path}")

    if "world_points" in predictions:
        print("Extracting 3D World Points...")
        # Shape is usually (B, S, H, W, 3) -> squeeze Batch dim
        wp = predictions["world_points"].cpu().float()
        if len(wp.shape) == 5:
            wp = wp.squeeze(0) 
    
        # wp shape is now (S, H, W, 3)
        colors = images.cpu().permute(0, 2, 3, 1) # -> (S, H, W, C)
        
        all_points = []
        all_colors = []
        
        for frame_idx in range(wp.shape[0]):
            points_3d = wp[frame_idx].reshape(-1, 3).numpy()
            point_colors = colors[frame_idx].reshape(-1, 3).numpy()
        
            # Optional: Filter out points with low confidence if available
            if "world_points_conf" in predictions:
                conf = predictions["world_points_conf"].cpu()
                if len(conf.shape) == 4:
                    conf = conf.squeeze(0)
                conf_frame = conf[frame_idx].reshape(-1)
                mask = conf_frame > 0.5 # Thresholding
                points_3d = points_3d[mask.numpy()]
                point_colors = point_colors[mask.numpy()]
                
            all_points.append(points_3d)
            all_colors.append(point_colors)
            
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

        import open3d as o3d
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
        # Save to disk
        pcd_path = os.path.join(out_dir, "scene_representation.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"3D Point Cloud saved to: {pcd_path}")
        
    if "pose_enc" in predictions:
        print("Extracting Camera Trajectory...")
        pose_enc = predictions["pose_enc"].cpu()
        if len(pose_enc.shape) == 2: # (S, 9)
            pose_enc = pose_enc.unsqueeze(0) # (1, S, 9)
            
        extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W), build_intrinsics=False)
        # extrinsics is (B, S, 3, 4), meaning R = (B,S,3,3), T = (B,S,3)
        R = extrinsics[0, :, :3, :3]
        T = extrinsics[0, :, :3, 3]
        
        # Invert to get camera centers in world coordinates: C = - R^T * T
        # R^T is transpose of R. So R.transpose(1, 2)
        R_inv = R.transpose(1, 2)
        T_inv = -torch.bmm(R_inv, T.unsqueeze(-1)).squeeze(-1)
        
        cam_trajectory = T_inv.numpy()
        traj_path = os.path.join(out_dir, "camera_trajectory.npy")
        np.save(traj_path, cam_trajectory)
        print(f"Camera trajectory saved to: {traj_path}")

    print(f"Done! Check the {out_dir} for all visualizations.")

if __name__ == "__main__":
    main()
