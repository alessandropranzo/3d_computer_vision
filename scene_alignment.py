import os
import open3d as o3d
import numpy as np
import torch

from vggt.utils.load_fn import load_and_preprocess_images
from extract_visualizations import generate_predictions

import argparse


def preprocess_cloud(points, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd


def align_clouds_icp(wp1, wp2, voxel_size=0.05):
    
    # 1. Convert numpy arrays to Open3D PointClouds
    source = preprocess_cloud(wp1, voxel_size)
    target = preprocess_cloud(wp2, voxel_size)

    # Set a threshold for ICP
    threshold = voxel_size * 10

    # Coarse alignment with FPFH
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    print(f"RANSAC fitness: {result_ransac.fitness:.3f}, RMSE: {result_ransac.inlier_rmse:.4f}")

    # 3. Run Point-to-Point ICP
    print("\nRunning ICP...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"ICP fitness: {reg_p2p.fitness:.3f}, RMSE: {reg_p2p.inlier_rmse:.4f}")

    if reg_p2p.fitness < 0.3:
        print("WARNING: Low ICP fitness — clouds may be in incompatible frames.")

    return reg_p2p.transformation


def main():
    parser = argparse.ArgumentParser(description="Align two point clouds")
    parser.add_argument("--path1", type=str, default="./data/images/room1/", help="Path to first image folder")
    parser.add_argument("--path2", type=str, default="./data/images/room2/", help="Path to second image folder")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for ICP")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    
    image_names1 = [os.path.join(args.path1, f) for f in sorted(os.listdir(args.path1)) if os.path.isfile(os.path.join(args.path1, f))]
    image_names2 = [os.path.join(args.path2, f) for f in sorted(os.listdir(args.path2)) if os.path.isfile(os.path.join(args.path2, f))]
    image_names = image_names1 + image_names2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Load images for VGGT
    images = load_and_preprocess_images(image_names).to(device)

    # Output of the VGGT model
    predictions = generate_predictions(images, device, dtype)

    wp = predictions["world_points"].squeeze(0).cpu().float().numpy()
    wp1 = wp[:len(image_names1)].reshape(-1, 3)
    wp2 = wp[len(image_names1):].reshape(-1, 3)

    colors = images.cpu().permute(0, 2, 3, 1).numpy()
    colors1 = colors[:len(image_names1)].reshape(-1, 3)
    colors2 = colors[len(image_names1):].reshape(-1, 3)

    # Find a transformation to align the two point clouds
    T = align_clouds_icp(wp1, wp2, voxel_size=args.voxel_size)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(wp1)
    source_pcd.colors = o3d.utility.Vector3dVector(colors1)
    source_pcd.transform(T)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(wp2)
    target_pcd.colors = o3d.utility.Vector3dVector(colors2)

    merged = source_pcd + target_pcd
    merged = merged.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)[0]

    o3d.io.write_point_cloud(os.path.join(args.out_dir, "aligned_scene_representation.ply"), merged)
    print(f"\nSaved aligned_scene_representation.ply")

if __name__ == "__main__":
    main()
