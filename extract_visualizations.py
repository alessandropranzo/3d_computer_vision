import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.visual_track import visualize_tracks_on_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import cv2
import argparse
from torchvision import transforms

def get_top_k_matching_pairs(dino_model, images1, images2, device, K=5):
    """Refine Part A: Extract DINOv2 features and find Top K matching pairs using Cosine Similarity."""
    print("Extracting DINOv2 features for cross-video matching...")
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        feat1 = dino_model.forward_features(transform(images1))['x_norm_clstoken'] # (len1, feat_dim)
        feat2 = dino_model.forward_features(transform(images2))['x_norm_clstoken'] # (len2, feat_dim)
        
        feat1 = torch.nn.functional.normalize(feat1, p=2, dim=-1)
        feat2 = torch.nn.functional.normalize(feat2, p=2, dim=-1)
        
        sim_matrix = torch.matmul(feat1, feat2.T) # (len1, len2)
        
        actual_K = min(K, len(images1) * len(images2))
        top_scores, top_indices = torch.topk(sim_matrix.flatten(), actual_K)
        
        pairs = []
        for i in range(actual_K):
            idx = top_indices[i].item()
            idx1 = idx // len(images2)
            idx2 = idx % len(images2)
            pairs.append((idx1, idx2))
            print(f"  Match {i+1}: Vid1:{idx1} Vid2:{idx2} Score: {top_scores[i].item():.4f}")
        return pairs

def validate_fusion(predictions, top_k_pairs, len_vid1, S, H, W, args, perm=None):
    """Refine Part B & C: Bridge Point Identification and RANSAC Validation."""
    print("\n--- Starting Geometric Validation ---")
    if "track" not in predictions or "world_points_conf" not in predictions:
        print("Required predictions (track/conf) missing for validation.")
        return False

    tracks = predictions["track"].cpu() # (B, S, N, 2)
    if len(tracks.shape) == 4:
        tracks = tracks.squeeze(0) # (S, N, 2)
        
    wp_conf = predictions["world_points_conf"].cpu() # (B, S, H, W)
    if len(wp_conf.shape) == 4:
        wp_conf = wp_conf.squeeze(0) # (S, H, W)
        
    total_inliers = 0
    min_inliers_required = 30 
    
    for p_idx, (idx1, idx2) in enumerate(top_k_pairs):
        # Map original indices to internal sequence indices based on shuffle
        m1, m2 = idx1, len_vid1 + idx2
            
        pts1 = tracks[m1] # (N, 2)
        pts2 = tracks[m2] # (N, 2)
        
        # Bridge Points: High confidence in both matching frames
        y1 = torch.clamp(pts1[:, 1].long(), 0, H-1)
        x1 = torch.clamp(pts1[:, 0].long(), 0, W-1)
        conf1 = wp_conf[m1, y1, x1]
        
        y2 = torch.clamp(pts2[:, 1].long(), 0, H-1)
        x2 = torch.clamp(pts2[:, 0].long(), 0, W-1)
        conf2 = wp_conf[m2, y2, x2]
        
        bridge_mask = (conf1 > 0.5) & (conf2 > 0.5)
        N_bridge = bridge_mask.sum().item()
        
        if N_bridge >= 8:
            pts1_v, pts2_v = pts1[bridge_mask].numpy(), pts2[bridge_mask].numpy()
            _, mask = cv2.findFundamentalMat(pts1_v, pts2_v, cv2.FM_RANSAC, 3.0, 0.99)
            if mask is not None:
                cnt = mask.sum()
                total_inliers += cnt
                print(f"  Match {p_idx+1}: {N_bridge} bridge pts -> {cnt} inliers.")
            else:
                print(f"  Match {p_idx+1}: RANSAC failed.")
        else:
            print(f"  Match {p_idx+1}: Insufficient bridge pts ({N_bridge}/8).")
            
    print(f"\nFinal Validation: {total_inliers} total inliers.")
    if total_inliers < min_inliers_required:
        print(f"RESULT: Fusion REJECTED (Required {min_inliers_required})")
        return False
    
    print(f"RESULT: Fusion ACCEPTED")
    return True

def plot_depth_maps(predictions, out_dir, S):
    """Save a grid of depth map visualizations."""
    if "depth" not in predictions: return
    print("Saving depth maps...")
    depth = predictions["depth"].cpu()
    # Handle various shape possibilities from VGGT
    if len(depth.shape) == 5: depth = depth.squeeze(0).squeeze(-1)
    elif len(depth.shape) == 4 and depth.shape[-1] == 1: depth = depth.squeeze(-1)
    else: depth = depth.squeeze(0)
    
    cols = 4
    rows = (S + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, ax in enumerate(axes_list):
        if i < S:
            im = ax.imshow(depth[i].numpy(), cmap='plasma')
            ax.set_title(f"Frame {i:02d}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "depth_grid.png"), dpi=150)
    plt.close()
    print(f"  Saved depth grid to {out_dir}/depth_grid.png")

def export_3d_results(predictions, images, out_dir, H, W):
    """Generate and save Point Cloud and Camera Trajectory."""
    if "world_points" in predictions:
        print("\nGenerating 3D Scene Representation (PLY)...")
        wp = predictions["world_points"].cpu().float().squeeze(0) if len(predictions["world_points"].shape) == 5 else predictions["world_points"].cpu().float()
        colors = images.cpu().permute(0, 2, 3, 1) 
        
        conf_all = None
        if "world_points_conf" in predictions:
            conf_all = predictions["world_points_conf"].cpu().float()
            if len(conf_all.shape) == 4: conf_all = conf_all.squeeze(0)

        all_pts, all_cols = [], []
        for i in range(wp.shape[0]):
            pts, cols = wp[i].reshape(-1, 3).numpy(), colors[i].reshape(-1, 3).numpy()
            if conf_all is not None:
                mask = (conf_all[i].reshape(-1) > 0.5).numpy()
                pts, cols = pts[mask], cols[mask]
            all_pts.append(pts)
            all_cols.append(cols)
            
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_pts, axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_cols, axis=0))
        o3d.io.write_point_cloud(os.path.join(out_dir, "scene_representation.ply"), pcd)
        print(f"  Saved PLY to {out_dir}/scene_representation.ply")

    if "pose_enc" in predictions:
        print("Extracting Camera Trajectory (NPY)...")
        pose_enc = predictions["pose_enc"].cpu()
        if len(pose_enc.shape) == 2: pose_enc = pose_enc.unsqueeze(0)
        extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W), build_intrinsics=False)
        R = extrinsics[0, :, :3, :3]
        T = extrinsics[0, :, :3, 3]
        R_inv = R.transpose(1, 2)
        T_inv = -torch.bmm(R_inv, T.unsqueeze(-1)).squeeze(-1)
        np.save(os.path.join(out_dir, "camera_trajectory.npy"), T_inv.numpy())
        print(f"  Saved NPY to {out_dir}/camera_trajectory.npy")

def main():
    parser = argparse.ArgumentParser(description="Extract Visualizations from VGGT")
    parser.add_argument("--path1", type=str, default="./data/images/room1/", help="Path to first image folder")
    parser.add_argument("--path2", type=str, default=None, help="Path to second image folder")
    parser.add_argument("--out_dir", type=str, default="predictions_visuals")
    args = parser.parse_args()

    # Hardware Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16 if device == "cuda" else torch.float32

    # Model Loading
    print(f"Loading Models on {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
    
    # Image Loading
    image_names1 = [os.path.join(args.path1, f) for f in sorted(os.listdir(args.path1)) if os.path.isfile(os.path.join(args.path1, f))]
    images1 = load_and_preprocess_images(image_names1).to(device)
    images = images1
    top_k_pairs = []
    len_vid1 = len(images1)
    perm = None

    if args.path2 is not None:
        image_names2 = [os.path.join(args.path2, f) for f in sorted(os.listdir(args.path2)) if os.path.isfile(os.path.join(args.path2, f))]
        images2 = load_and_preprocess_images(image_names2).to(device)
        top_k_pairs = get_top_k_matching_pairs(dino_model, images1, images2, device)
        
        min_len = min(len(images1), len(images2))
        alt_images = []
        for i in range(min_len):
            alt_images.append(images1[i:i+1]); alt_images.append(images2[i:i+1])
        if len(images1) > min_len: alt_images.append(images1[min_len:])
        if len(images2) > min_len: alt_images.append(images2[min_len:])
        images = torch.cat(alt_images, dim=0)

    # Inference
    S, C, H, W = images.shape
    grid_y, grid_x = torch.meshgrid(torch.linspace(20, H-20, steps=30), torch.linspace(20, W-20, steps=30), indexing='ij')
    query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(device)
    
    print(f"Running VGGT inference on {S} frames...")
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.no_grad():
            predictions = model(images, query_points=query_points)
            
    os.makedirs(args.out_dir, exist_ok=True)

    # 2D Visualizations
    plot_depth_maps(predictions, args.out_dir, S)

    # Geometric Validation & 3D Export
    fusion_accepted = True
    if args.path2 is not None:
        fusion_accepted = validate_fusion(predictions, top_k_pairs, len_vid1, S, H, W, args, perm)

    if fusion_accepted:
        export_3d_results(predictions, images, args.out_dir, H, W)
    else:
        print("Skipping 3D reconstruction due to fusion rejection.")

    print(f"\nDone! Results in {args.out_dir}")

if __name__ == "__main__":
    main()
