import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.visual_track import visualize_tracks_on_images

def main():
    path = "./data/images_sampled_down/"
    image_names = [os.path.join(path, f) for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    image_names = image_names[::2]  # Subsample frames to reduce memory
    
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
    
    print(f"Loading {len(image_names)} images...")
    images = load_and_preprocess_images(image_names).to(device)
    
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
            
    out_dir = "predictions_visuals"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Plot Tracks
    if "track" in predictions:
        print("Plotting tracks...")
        track = predictions["track"]
        if "vis" in predictions:
            vis = predictions["vis"] > 0.5
        else:
            vis = None
            
        # Use images returned by model (might have proper scaling and shape)
        images_to_plot = predictions.get("images", images)
        track_out = os.path.join(out_dir, "tracks")
        os.makedirs(track_out, exist_ok=True)
        
        visualize_tracks_on_images(
            images=images_to_plot,
            tracks=track,
            track_vis_mask=vis,
            out_dir=track_out,
            image_format="CHW",
            normalize_mode="[0,1]",
            frames_per_row=4,
            save_grid=True
        )
        print(f"Tracks grid saved to: {track_out}/tracks_grid.png")
    else:
        print("Warning: No 'track' key found in predictions.")
        
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
        # Let's extract points from the first frame as an example
        frame_idx = 0
        points_3d = wp[frame_idx].reshape(-1, 3).numpy()
    
        # Optional: Filter out points with low confidence if available
        if "world_points_conf" in predictions:
            conf = predictions["world_points_conf"].cpu()
            if len(conf.shape) == 4:
                conf = conf.squeeze(0)
            conf_frame = conf[frame_idx].reshape(-1)
            mask = conf_frame > 0.5 # Thresholding
            points_3d = points_3d[mask.numpy()]

            import open3d as o3d
            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
        
            # Save to disk
            pcd_path = os.path.join(out_dir, "scene_representation.ply")
            o3d.io.write_point_cloud(pcd_path, pcd)
            print(f"3D Point Cloud saved to: {pcd_path}")
    
    print(f"Done! Check the {out_dir} for all visualizations.")

if __name__ == "__main__":
    main()
