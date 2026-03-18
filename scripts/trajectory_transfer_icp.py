#!/usr/bin/env python3
"""
Trajectory Transfer via ICP
===========================
Compute novel camera poses by aligning two independent 3D representations
(point clouds from VGGT) using ICP, and applying the computed transformation
to the camera trajectory of the second room.

Stages:
  1. Load point clouds (scene_representation.ply) and camera parameters (extrinsics.npy)
     for Room 1 and Room 2.
  2. Perform global registration followed by ICP to align Room 2 to Room 1.
  3. Extract the 4x4 transformation matrix (rotation and translation).
  4. Apply the transformation to Room 2's camera trajectory to bring it into Room 1's 3D space.
  5. Save the transferred novel poses.

Usage:
    python trajectory_transfer_icp.py \
        --room1_dir predictions_visuals/room1 \
        --room2_dir predictions_visuals/room2 \
        --output_dir transfer_results_icp
"""

import os
import argparse
import copy
import numpy as np
import open3d as o3d

# ============================================================
# Data Loading
# ============================================================

def load_predictions(pred_dir):
    """Load camera data from a VGGT predictions directory."""
    data = {}
    for name in ["extrinsics", "intrinsics"]:
        path = os.path.join(pred_dir, f"{name}.npy")
        if os.path.exists(path):
            data[name] = np.load(path)
        else:
            raise FileNotFoundError(f"Missing {path}")
    return data

def load_point_cloud(pred_dir):
    """Load the point cloud saved by extract_visualizations.py."""
    path = os.path.join(pred_dir, "scene_representation.ply")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Please run extract_visualizations.py first.")
    pcd = o3d.io.read_point_cloud(path)
    return pcd

# ============================================================
# Point Cloud Registration (Global + ICP)
# ============================================================

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample and compute features for global registration."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Global registration using RANSAC to get an initial alignment."""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def align_point_clouds(source, target, voxel_size=0.05):
    """Align source to target using Global Registration followed by ICP."""
    # Preprocess
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    # Global Registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    
    # Local Registration (ICP)
    distance_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return result_icp.transformation, result_icp

# ============================================================
# Trajectory Transfer
# ============================================================

def transform_extrinsics(extrinsics, transformation_matrix):
    """
    Apply a 4x4 transformation matrix (which transforms points from source to target)
    to camera extrinsics [R|t].
    
    If transformation matrix is T = [R_T | t_T], 
    points are transformed as: x_w1 = R_T * x_w2 + t_T
    
    For a camera in source space:
    x_c = R_2 * x_w2 + t_2
    
    Substituting x_w2 = R_T.T * (x_w1 - t_T)
    x_c = R_2 * R_T.T * x_w1 + t_2 - R_2 * R_T.T * t_T
    
    Therefore, novel extrinsics in target space:
    R_novel = R_2 @ R_T.T
    t_novel = t_2 - R_2 @ R_T.T @ t_T
    """
    R_T = transformation_matrix[:3, :3]
    t_T = transformation_matrix[:3, 3]
    
    N = extrinsics.shape[0]
    novel_extrinsics = np.zeros_like(extrinsics)
    
    for i in range(N):
        R_2 = extrinsics[i, :3, :3]
        t_2 = extrinsics[i, :3, 3]
        
        R_novel = R_2 @ R_T.T
        t_novel = t_2 - R_novel @ t_T
        
        novel_extrinsics[i, :3, :3] = R_novel
        novel_extrinsics[i, :3, 3] = t_novel
        
    return novel_extrinsics

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trajectory Transfer via ICP - align two rooms and transfer trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--room1_dir", default="results/vggt_predictions/room1", help="Path to target room predictions (Room 1)")
    parser.add_argument("--room2_dir", default="results/vggt_predictions/room2", help="Path to source room predictions (Room 2)")
    parser.add_argument("--output_dir", default="results/trajectory_transfers/icp", help="Path to save results")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for downsampling and ICP")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 1 - Loading data")
    print("=" * 60)
    
    print(f"Loading Room 1 from {args.room1_dir}...")
    room1_data = load_predictions(args.room1_dir)
    pcd_room1 = load_point_cloud(args.room1_dir)
    
    print(f"Loading Room 2 from {args.room2_dir}...")
    room2_data = load_predictions(args.room2_dir)
    pcd_room2 = load_point_cloud(args.room2_dir)

    print(f"Room 1: {len(pcd_room1.points)} points, {room1_data['extrinsics'].shape[0]} cameras")
    print(f"Room 2: {len(pcd_room2.points)} points, {room2_data['extrinsics'].shape[0]} cameras")
    
    print("\n" + "=" * 60)
    print("Stage 2 - Aligning point clouds (Room 2 -> Room 1)")
    print("=" * 60)
    
    transformation_matrix, result_icp = align_point_clouds(pcd_room2, pcd_room1, voxel_size=args.voxel_size)
    print("ICP Alignment Result:")
    print(result_icp)
    print("Transformation Matrix:")
    print(transformation_matrix)
    
    np.savez(
        os.path.join(args.output_dir, "alignment.npz"),
        transformation=transformation_matrix,
        rotation=transformation_matrix[:3, :3],
        translation=transformation_matrix[:3, 3]
    )

    # Optional: Save aligned point cloud for visualization
    pcd_room2_aligned = copy.deepcopy(pcd_room2).transform(transformation_matrix)
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "scene_representation_room2_aligned.ply"), pcd_room2_aligned)

    print("\n" + "=" * 60)
    print("Stage 3 - Transferring trajectory")
    print("=" * 60)
    
    extri_room2 = room2_data["extrinsics"]
    intri_room2 = room2_data["intrinsics"]
    
    novel_extri = transform_extrinsics(extri_room2, transformation_matrix)
    
    np.save(os.path.join(args.output_dir, "novel_extrinsics.npy"), novel_extri)
    np.save(os.path.join(args.output_dir, "novel_intrinsics.npy"), intri_room2) # Intrinsics remain identical
    
    # In order to match the output format of trajectory_transfer.py to be compatible 
    # with the rest of the pipeline which might look for match_row_ind.npy and match_col_ind.npy
    n_poses = novel_extri.shape[0]
    n_room1 = room1_data["extrinsics"].shape[0]
    
    # We use linear mappings but clip them to avoid IndexErrors in rendering scripts
    dummy_row = np.minimum(np.arange(n_poses), n_room1 - 1)
    dummy_col = np.arange(n_poses)
    
    np.save(os.path.join(args.output_dir, "match_row_ind.npy"), dummy_row)
    np.save(os.path.join(args.output_dir, "match_col_ind.npy"), dummy_col)

    print(f"  {novel_extri.shape[0]} novel poses saved")
    print(f"\nDone! Results saved in {args.output_dir}/")

if __name__ == "__main__":
    main()
