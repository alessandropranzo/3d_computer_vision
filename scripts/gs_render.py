#!/usr/bin/env python3
"""
Gaussian Splatting Renderer
===========================
Renders the transferred camera trajectory using a trained 3DGS model.

Supports two input formats:
  1. Standard 3DGS PLY: e.g. point_cloud/iteration_30000/point_cloud.ply
  2. CF-3DGS Checkpoint: e.g. chkpnt/ep00_init.pth

Usage:
    # First, make sure you are in an environment with diff_gaussian_rasterization installed.
    
    python gs_render.py \\
        --model_path output/gs_room1/point_cloud/iteration_30000/point_cloud.ply \\
        --transfer_dir transfer_results \\
        --images_room1 data/images/room1 \\
        --images_room2 data/images/room2 \\
        --output_dir gs_results

    # For CF-3DGS:
    python gs_render.py \\
        --model_path output/progressive/my_room/chkpnt/ep00_init.pth \\
        ...
"""

import os
import math
import argparse
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    print("Error: diff_gaussian_rasterization is not installed.")
    print("Please install it from: https://github.com/graphdeco-inria/diff-gaussian-rasterization")
    sys.exit(1)


# ============================================================
# Model Loading
# ============================================================

def load_ply(path):
    from plyfile import PlyData
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # Attempt to load rest features if available
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    if len(extra_f_names) > 0:
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))
    else:
        features_extra = np.zeros((xyz.shape[0], 3, 15))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Convert to torch
    return {
        "xyz": torch.tensor(xyz, dtype=torch.float32, device="cuda"),
        "features_dc": torch.tensor(features_dc, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous(),
        "features_rest": torch.tensor(features_extra, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous(),
        "opacity": torch.tensor(opacities, dtype=torch.float32, device="cuda"),
        "scaling": torch.tensor(scales, dtype=torch.float32, device="cuda"),
        "rotation": torch.tensor(rots, dtype=torch.float32, device="cuda"),
        "active_sh_degree": 3 if len(extra_f_names) > 0 else 0
    }

def load_checkpoint(path):
    data = torch.load(path, map_location="cuda")
    if isinstance(data, tuple) and len(data) >= 7:
        active_sh_degree = data[0]
        xyz = data[1]
        features_dc = data[2]
        features_rest = data[3]
        scaling = data[4]
        rotation = data[5]
        opacity = data[6]
        
        # Ensure features are in the right shape: (N, num_sh, 3)
        if features_dc.dim() == 2:
            features_dc = features_dc.unsqueeze(1)
        if features_rest.dim() == 2:
            features_rest = features_rest.view(features_rest.shape[0], -1, 3)

        return {
            "xyz": xyz,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "active_sh_degree": active_sh_degree
        }
    else:
        raise ValueError("Unrecognized checkpoint format.")


# ============================================================
# Camera Utils
# ============================================================

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# ============================================================
# Rendering
# ============================================================

def render_frame(gs_data, extrinsic, intrinsic, H, W, bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")):
    # Parse camera
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    
    # 3DGS expects column-major R
    R_col_major = R.transpose()
    world_view_transform = torch.eye(4, dtype=torch.float32, device="cuda")
    world_view_transform[:3, :3] = torch.tensor(R_col_major, dtype=torch.float32, device="cuda")
    world_view_transform[3, :3] = torch.tensor(T, dtype=torch.float32, device="cuda")
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    fovX = focal2fov(fx, W)
    fovY = focal2fov(fy, H)
    
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovX, fovY=fovY).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(H),
        image_width=int(W),
        tanfovx=math.tan(fovX * 0.5),
        tanfovy=math.tan(fovY * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=gs_data["active_sh_degree"],
        campos=camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Render
    means3D = gs_data["xyz"]
    means2D = torch.zeros_like(means3D, requires_grad=False)
    opacities = torch.sigmoid(gs_data["opacity"])
    scales = torch.exp(gs_data["scaling"])
    rotations = torch.nn.functional.normalize(gs_data["rotation"], dim=-1)
    shs = torch.cat([gs_data["features_dc"], gs_data["features_rest"]], dim=1)

    rendered_image, radii, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    return rendered_image.clamp(0, 1)

def main():
    parser = argparse.ArgumentParser(description="Render 3DGS model on transferred trajectory")
    parser.add_argument("--model_path", required=True, help="Path to .ply or .pth model")
    parser.add_argument("--transfer_dir", default="results/trajectory_transfers/base")
    parser.add_argument("--images_room1", default="data/images/room1")
    parser.add_argument("--images_room2", default="data/images/room2")
    parser.add_argument("--output_dir", default="results/gs_renders/base")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading 3DGS model...")
    if args.model_path.endswith(".ply"):
        gs_data = load_ply(args.model_path)
    elif args.model_path.endswith(".pth"):
        gs_data = load_checkpoint(args.model_path)
    else:
        print("Unknown model format. Must be .ply or .pth")
        sys.exit(1)
        
    print(f"Loaded {gs_data['xyz'].shape[0]} gaussians.")

    print("Loading novel trajectory...")
    novel_extri = np.load(os.path.join(args.transfer_dir, "novel_extrinsics.npy"))
    novel_intri = np.load(os.path.join(args.transfer_dir, "novel_intrinsics.npy"))
    row_ind = np.load(os.path.join(args.transfer_dir, "match_row_ind.npy"))
    col_ind = np.load(os.path.join(args.transfer_dir, "match_col_ind.npy"))

    # Load reference images to get dimensions and for comparisons
    fnames = sorted([f for f in os.listdir(args.images_room1) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    first_img = Image.open(os.path.join(args.images_room1, fnames[0]))
    W, H = first_img.size
    
    target_width = 518
    target_height = round(H * (target_width / W) / 14) * 14
    if target_height > target_width:
        start_y = (target_height - target_width) // 2
        crop_box = (0, start_y, target_width, start_y + target_width)
        W_out, H_out = target_width, target_width
    else:
        crop_box = None
        W_out, H_out = target_width, target_height
        
    def load_cropped_image(path):
        img = Image.open(path).convert("RGB")
        img = img.resize((target_width, target_height), Image.BICUBIC)
        if crop_box is not None:
            img = img.crop(crop_box)
        return np.asarray(img) / 255.0

    images_r1 = [load_cropped_image(os.path.join(args.images_room1, f)) for f in fnames]
    
    images_r2 = []
    if os.path.exists(args.images_room2):
        fnames2 = sorted([f for f in os.listdir(args.images_room2) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        images_r2 = [load_cropped_image(os.path.join(args.images_room2, f)) for f in fnames2]

    render_dir = os.path.join(args.output_dir, "transferred_views")
    cmp_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(cmp_dir, exist_ok=True)

    print(f"\nRendering {novel_extri.shape[0]} frames...")
    
    n_cols = 3 if len(images_r2) > 0 else 2
    
    with torch.no_grad():
        for i in tqdm(range(novel_extri.shape[0])):
            ext = novel_extri[i]
            intri = novel_intri[i]
            
            rendered_ts = render_frame(gs_data, ext, intri, H_out, W_out)
            rendered_np = rendered_ts.permute(1, 2, 0).cpu().numpy()
            
            plt.imsave(os.path.join(render_dir, f"transferred_{i:04d}.png"), rendered_np)
            
            # Plot comparisons
            fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
            
            axes[0].imshow(images_r1[row_ind[i]])
            axes[0].set_title(f"Room 1 – frame {row_ind[i]}")
            axes[0].axis("off")

            axes[1].imshow(rendered_np)
            axes[1].set_title(f"Transferred view {i}")
            axes[1].axis("off")

            if len(images_r2) > 0 and col_ind[i] < len(images_r2):
                axes[2].imshow(images_r2[col_ind[i]])
                axes[2].set_title(f"Room 2 ref – frame {col_ind[i]}")
                axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(cmp_dir, f"compare_{i:04d}.png"), dpi=150)
            plt.close()

    print(f"\nDone! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
