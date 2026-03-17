#!/usr/bin/env python3
"""
NeRF Transfer Pipeline
======================
1. Load camera parameters from VGGT predictions (room1-only and merged).
2. Match cameras between rooms in merged space (Hungarian algorithm).
3. Compute trajectory difference (position + rotation deltas).
4. Procrustes alignment between merged and standalone coordinate systems.
5. Train a NeRF on room 1 images + standalone camera poses.
6. Compute novel camera poses and render transferred views.

Usage:
    python nerf_transfer_pipeline.py \
        --merged_dir predictions_visuals/merged \
        --room1_dir predictions_visuals/room1 \
        --images_room1 data/images/room1 \
        --images_room2 data/images/room2
"""

import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# Data Loading
# ============================================================

def load_predictions(pred_dir):
    """Load all .npy camera data from a VGGT predictions directory."""
    data = {}
    for name in [
        "extrinsics", "intrinsics", "camera_trajectory_merged",
        "extrinsics_v1", "extrinsics_v2", "intrinsics_v1", "intrinsics_v2",
        "camera_trajectory_v1", "camera_trajectory_v2",
    ]:
        path = os.path.join(pred_dir, f"{name}.npy")
        if os.path.exists(path):
            data[name] = np.load(path)
    return data


def load_images_for_nerf(image_dir, target_width=518):
    """Load images with VGGT-compatible preprocessing (crop mode).

    Resizes width to 518, adjusts height to be divisible by 14,
    and center-crops if height exceeds 518.  Returns (N, H, W, 3) float32 in [0, 1].
    """
    fnames = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    images = []
    for fname in fnames:
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        w, h = img.size
        new_w = target_width
        new_h = round(h * (new_w / w) / 14) * 14
        img = img.resize((new_w, new_h), Image.BICUBIC)
        if new_h > target_width:
            start_y = (new_h - target_width) // 2
            img = img.crop((0, start_y, new_w, start_y + target_width))
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(images)


def extrinsics_to_centers(extrinsics):
    """(N, 3, 4) extrinsics [R|t] -> (N, 3) camera centres.  C = -R^T t."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)


# ============================================================
# Camera Matching & Trajectory Difference
# ============================================================

def match_cameras_hungarian(centers1, centers2):
    """Optimal one-to-one matching by Euclidean distance (Hungarian).

    Returns (row_ind, col_ind) arrays of matched indices.
    len(row_ind) == min(len(centers1), len(centers2)).
    """
    dist = np.linalg.norm(centers1[:, None] - centers2[None, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(dist)
    return row_ind, col_ind


def compute_trajectory_deltas(extrinsics1, extrinsics2, row_ind, col_ind):
    """Position and rotation deltas between matched pairs.

    Returns:
        delta_pos  (M, 3)   – C2[col] - C1[row]
        delta_rot  (M, 3, 3) – R2 @ R1^T  (so that R2 = delta_R @ R1)
    """
    c1 = extrinsics_to_centers(extrinsics1[row_ind])
    c2 = extrinsics_to_centers(extrinsics2[col_ind])
    delta_pos = c2 - c1

    R1 = extrinsics1[row_ind, :3, :3]
    R2 = extrinsics2[col_ind, :3, :3]
    delta_rot = np.einsum("nij,nkj->nik", R2, R1)  # R2 @ R1^T
    return delta_pos, delta_rot


# ============================================================
# Procrustes Alignment
# ============================================================

def procrustes_alignment(source, target):
    """SVD-based Procrustes: target ≈ s * R @ source + t.

    Returns (scale, R_3x3, t_3).
    """
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src_c = source - mu_s
    tgt_c = target - mu_t

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, d])
    R = Vt.T @ S @ U.T

    scale = np.trace(R @ H) / np.trace(src_c.T @ src_c)
    t = mu_t - scale * R @ mu_s
    return scale, R, t


def apply_alignment_to_deltas(deltas, scale, rotation):
    """Rotate and scale delta vectors into the target coordinate system."""
    return scale * (rotation @ deltas.T).T


# ============================================================
# NeRF Components
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.register_buffer(
            "freq_bands", 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        )

    def forward(self, x):
        parts = [x] if self.include_input else []
        for freq in self.freq_bands:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)

    def out_dim(self, in_dim):
        return in_dim * (1 + 2 * self.num_freqs) if self.include_input else in_dim * 2 * self.num_freqs


class NeRFMLP(nn.Module):
    """8-layer MLP with skip connection at layer 4 (original NeRF design)."""

    def __init__(self, pos_freqs=10, dir_freqs=4, hidden=256):
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_freqs)
        self.dir_enc = PositionalEncoding(dir_freqs)
        pos_dim = self.pos_enc.out_dim(3)
        dir_dim = self.dir_enc.out_dim(3)

        self.block1 = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden + pos_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.sigma_head = nn.Linear(hidden, 1)
        self.feature_head = nn.Linear(hidden, hidden)
        self.color_block = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, pos, view_dir):
        """pos: (..., 3), view_dir: (..., 3) -> rgb (..., 3), sigma (..., 1)"""
        pe = self.pos_enc(pos)
        de = self.dir_enc(view_dir)
        h = self.block1(pe)
        h = self.block2(torch.cat([h, pe], dim=-1))
        sigma = F.softplus(self.sigma_head(h))
        feat = self.feature_head(h)
        rgb = self.color_block(torch.cat([feat, de], dim=-1))
        return rgb, sigma


# ---- Ray helpers --------------------------------------------------------

def get_rays_np(H, W, K, extrinsic):
    """All-pixel rays for one camera (numpy).

    K:         (3, 3) intrinsic
    extrinsic: (3, 4) [R|t]  (camera-from-world, OpenCV convention)
    Returns origins (H*W, 3), directions (H*W, 3) – world space, unit-norm dirs.
    """
    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32) + 0.5,
        np.arange(H, dtype=np.float32) + 0.5,
    )
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(H * W, dtype=np.float32)], axis=-1)
    K_inv = np.linalg.inv(K)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    dirs_cam = (K_inv @ pixels.T).T
    dirs_world = (R.T @ dirs_cam.T).T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    origin = -R.T @ t
    origins = np.broadcast_to(origin, dirs_world.shape).copy()
    return origins, dirs_world


# ---- Sampling -----------------------------------------------------------

def stratified_sample(rays_o, rays_d, near, far, N, perturb=True):
    """Stratified sampling.  Returns pts (B, N, 3), t_vals (B, N)."""
    B = rays_o.shape[0]
    t = torch.linspace(0.0, 1.0, N, device=rays_o.device)
    t = near + (far - near) * t
    t = t[None].expand(B, -1)
    if perturb:
        mid = 0.5 * (t[:, 1:] + t[:, :-1])
        upper = torch.cat([mid, t[:, -1:]], dim=-1)
        lower = torch.cat([t[:, :1], mid], dim=-1)
        t = lower + (upper - lower) * torch.rand_like(t)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t[..., None]
    return pts, t


def sample_pdf(bins, weights, N):
    """Inverse-CDF sampling from a piecewise-constant PDF.

    bins:    (B, M)     – bin edges
    weights: (B, M-1)   – unnormalised bin weights
    N:       int         – number of samples
    Returns  (B, N)      – sampled t-values.
    """
    w = weights + 1e-5
    pdf = w / w.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    u = torch.rand(weights.shape[0], N, device=weights.device).contiguous()
    idx = torch.searchsorted(cdf.contiguous(), u, right=True)
    lo = (idx - 1).clamp(min=0)
    hi = idx.clamp(max=cdf.shape[-1] - 1)

    cdf_lo = torch.gather(cdf, 1, lo)
    cdf_hi = torch.gather(cdf, 1, hi)
    bins_lo = torch.gather(bins, 1, lo)
    bins_hi = torch.gather(bins, 1, hi)

    denom = (cdf_hi - cdf_lo).clamp(min=1e-5)
    t = (u - cdf_lo) / denom
    return bins_lo + t * (bins_hi - bins_lo)


# ---- Volume rendering ---------------------------------------------------

def volume_render(rgb, sigma, t_vals, rays_d):
    """Classic NeRF alpha-compositing.

    rgb:    (B, N, 3)
    sigma:  (B, N, 1)
    t_vals: (B, N)
    rays_d: (B, 3)
    Returns rendered (B, 3), weights (B, N), depth (B,), acc (B,).
    """
    sigma = sigma.squeeze(-1)
    dists = t_vals[:, 1:] - t_vals[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
    dists = dists * rays_d.norm(dim=-1, keepdim=True)

    alpha = 1.0 - torch.exp(-sigma * dists)
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=-1)

    weights = T * alpha
    rendered = (weights[..., None] * rgb).sum(dim=1)
    depth = (weights * t_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)
    return rendered, weights, depth, acc


# ============================================================
# Training
# ============================================================

def estimate_scene_bounds(extrinsics):
    """Heuristic near / far from camera baseline."""
    c = extrinsics_to_centers(extrinsics)
    d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    scale = max(d.max(), 0.1)
    return 0.02 * scale, 6.0 * scale


def train_nerf(
    images, extrinsics, intrinsics,
    num_iters=20000, batch_size=1024, lr=5e-4,
    N_coarse=64, N_fine=128, device="cuda",
):
    """Train coarse + fine NeRF on room-1 data.

    images:     (N, H, W, 3) float32 [0,1]
    extrinsics: (N, 3, 4)
    intrinsics: (N, 3, 3)
    """
    N_img, H, W, _ = images.shape
    near, far = estimate_scene_bounds(extrinsics)
    print(f"  Scene bounds: near={near:.4f}, far={far:.4f}")

    # Pre-compute all rays
    all_o, all_d, all_c = [], [], []
    for i in range(N_img):
        ro, rd = get_rays_np(H, W, intrinsics[i], extrinsics[i])
        all_o.append(ro)
        all_d.append(rd)
        all_c.append(images[i].reshape(-1, 3))

    all_o = torch.from_numpy(np.concatenate(all_o)).float().to(device)
    all_d = torch.from_numpy(np.concatenate(all_d)).float().to(device)
    all_c = torch.from_numpy(np.concatenate(all_c)).float().to(device)
    n_rays = all_o.shape[0]
    print(f"  Total training rays: {n_rays:,}")

    coarse = NeRFMLP().to(device)
    fine = NeRFMLP().to(device)
    opt = torch.optim.Adam(list(coarse.parameters()) + list(fine.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_iters)

    pbar = tqdm(range(1, num_iters + 1), desc="  NeRF training")
    for step in pbar:
        idx = torch.randint(0, n_rays, (batch_size,), device=device)
        ro, rd, tgt = all_o[idx], all_d[idx], all_c[idx]

        # --- coarse ---
        pts_c, t_c = stratified_sample(ro, rd, near, far, N_coarse, perturb=True)
        d_c = rd[:, None, :].expand_as(pts_c)
        rgb_c, sig_c = coarse(pts_c.reshape(-1, 3), d_c.reshape(-1, 3))
        rgb_c = rgb_c.view(batch_size, N_coarse, 3)
        sig_c = sig_c.view(batch_size, N_coarse, 1)
        rend_c, w_c, _, acc_c = volume_render(rgb_c, sig_c, t_c, rd)

        # --- fine (hierarchical) ---
        t_mid = 0.5 * (t_c[:, 1:] + t_c[:, :-1])
        t_f = sample_pdf(t_mid, w_c[:, 1:-1].detach(), N_fine)
        t_all, _ = torch.sort(torch.cat([t_c, t_f], dim=-1), dim=-1)
        N_all = N_coarse + N_fine

        pts_f = ro[:, None, :] + rd[:, None, :] * t_all[..., None]
        d_f = rd[:, None, :].expand_as(pts_f)
        rgb_f, sig_f = fine(pts_f.reshape(-1, 3), d_f.reshape(-1, 3))
        rgb_f = rgb_f.view(batch_size, N_all, 3)
        sig_f = sig_f.view(batch_size, N_all, 1)
        rend_f, _, _, acc_f = volume_render(rgb_f, sig_f, t_all, rd)

        # --- losses ---
        loss_c = F.mse_loss(rend_c, tgt)
        loss_f = F.mse_loss(rend_f, tgt)
        ent_c = -(acc_c * (acc_c + 1e-6).log() + (1 - acc_c) * (1 - acc_c + 1e-6).log()).mean()
        ent_f = -(acc_f * (acc_f + 1e-6).log() + (1 - acc_f) * (1 - acc_f + 1e-6).log()).mean()
        loss = loss_c + loss_f + 0.01 * (ent_c + ent_f)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 500 == 0:
            psnr = -10.0 * math.log10(max(loss_f.item(), 1e-10))
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.1f}")

    return coarse, fine, near, far


# ============================================================
# Rendering
# ============================================================

@torch.no_grad()
def render_image(coarse, fine, extrinsic, intrinsic, H, W, near, far,
                 N_coarse=64, N_fine=128, chunk=4096, device="cuda"):
    """Render a full image from a single camera pose."""
    ro, rd = get_rays_np(H, W, intrinsic, extrinsic)
    ro = torch.from_numpy(ro).float().to(device)
    rd = torch.from_numpy(rd).float().to(device)
    n = ro.shape[0]
    out = torch.zeros(n, 3, device=device)

    for i in range(0, n, chunk):
        ro_b = ro[i : i + chunk]
        rd_b = rd[i : i + chunk]
        bs = ro_b.shape[0]

        pts_c, t_c = stratified_sample(ro_b, rd_b, near, far, N_coarse, perturb=False)
        d_c = rd_b[:, None, :].expand_as(pts_c)
        rgb_c, sig_c = coarse(pts_c.reshape(-1, 3), d_c.reshape(-1, 3))
        rgb_c = rgb_c.view(bs, N_coarse, 3)
        sig_c = sig_c.view(bs, N_coarse, 1)
        _, w_c, _, _ = volume_render(rgb_c, sig_c, t_c, rd_b)

        t_mid = 0.5 * (t_c[:, 1:] + t_c[:, :-1])
        t_f = sample_pdf(t_mid, w_c[:, 1:-1], N_fine)
        t_all, _ = torch.sort(torch.cat([t_c, t_f], dim=-1), dim=-1)
        N_all = N_coarse + N_fine

        pts_f = ro_b[:, None, :] + rd_b[:, None, :] * t_all[..., None]
        d_f = rd_b[:, None, :].expand_as(pts_f)
        rgb_f, sig_f = fine(pts_f.reshape(-1, 3), d_f.reshape(-1, 3))
        rgb_f = rgb_f.view(bs, N_all, 3)
        sig_f = sig_f.view(bs, N_all, 1)
        rend, _, _, _ = volume_render(rgb_f, sig_f, t_all, rd_b)
        out[i : i + chunk] = rend

    return out.view(H, W, 3).cpu().numpy()


# ============================================================
# Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NeRF Transfer Pipeline")
    parser.add_argument("--merged_dir", default="predictions_visuals/merged")
    parser.add_argument("--room1_dir", default="predictions_visuals/room1")
    parser.add_argument("--images_room1", default="data/images/room1")
    parser.add_argument("--images_room2", default="data/images/room2")
    parser.add_argument("--output_dir", default="nerf_results")
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ==== Stage 1: Load predictions ====
    print("=" * 60)
    print("Stage 1 – Loading predictions")
    print("=" * 60)

    merged = load_predictions(args.merged_dir)
    room1 = load_predictions(args.room1_dir)

    extri_merged = merged["extrinsics"]   # (N_total, 3, 4)
    intri_merged = merged["intrinsics"]   # (N_total, 3, 3)
    extri_room1 = room1["extrinsics"]     # (N1, 3, 4)
    intri_room1 = room1["intrinsics"]     # (N1, 3, 3)

    n1 = len([
        f for f in sorted(os.listdir(args.images_room1))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    n_total = extri_merged.shape[0]
    n2 = n_total - n1

    extri_m_r1 = extri_merged[:n1]
    extri_m_r2 = extri_merged[n1:]
    centers_m_r1 = extrinsics_to_centers(extri_m_r1)
    centers_m_r2 = extrinsics_to_centers(extri_m_r2)
    centers_standalone = extrinsics_to_centers(extri_room1)

    print(f"  Merged frames : {n_total}  (room1={n1}, room2={n2})")
    print(f"  Standalone r1 : {extri_room1.shape[0]} frames")

    # ==== Stage 2: Camera matching ====
    print("\n" + "=" * 60)
    print("Stage 2 – Hungarian camera matching")
    print("=" * 60)

    row_ind, col_ind = match_cameras_hungarian(centers_m_r1, centers_m_r2)
    match_dists = np.linalg.norm(
        centers_m_r1[row_ind] - centers_m_r2[col_ind], axis=-1
    )
    print(f"  Matched {len(row_ind)} pairs")
    print(f"  Distances: mean={match_dists.mean():.4f}  "
          f"min={match_dists.min():.4f}  max={match_dists.max():.4f}")

    np.save(os.path.join(args.output_dir, "match_row_ind.npy"), row_ind)
    np.save(os.path.join(args.output_dir, "match_col_ind.npy"), col_ind)

    # ==== Stage 3: Trajectory difference ====
    print("\n" + "=" * 60)
    print("Stage 3 – Trajectory deltas")
    print("=" * 60)

    delta_pos, delta_rot = compute_trajectory_deltas(
        extri_m_r1, extri_m_r2, row_ind, col_ind
    )
    print(f"  Pos-delta mean norm : {np.linalg.norm(delta_pos, axis=-1).mean():.4f}")

    # ==== Stage 4: Procrustes alignment ====
    print("\n" + "=" * 60)
    print("Stage 4 – Procrustes alignment (merged → standalone)")
    print("=" * 60)

    scale, R_align, t_align = procrustes_alignment(centers_m_r1, centers_standalone)
    aligned = scale * (R_align @ centers_m_r1.T).T + t_align
    residual = np.linalg.norm(aligned - centers_standalone, axis=-1).mean()
    print(f"  Scale    : {scale:.6f}")
    print(f"  Residual : {residual:.6f}")

    delta_aligned = apply_alignment_to_deltas(delta_pos, scale, R_align)
    print(f"  Aligned-delta mean norm : {np.linalg.norm(delta_aligned, axis=-1).mean():.4f}")

    # ==== Stage 5: Train NeRF on room 1 ====
    print("\n" + "=" * 60)
    print("Stage 5 – Training NeRF on room 1")
    print("=" * 60)

    images = load_images_for_nerf(args.images_room1)
    H, W = images.shape[1], images.shape[2]
    print(f"  Images: {images.shape[0]} × {H}×{W}")

    coarse, fine, near, far = train_nerf(
        images, extri_room1, intri_room1,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        device=args.device,
    )

    ckpt_path = os.path.join(args.output_dir, "nerf_checkpoint.pt")
    torch.save({
        "coarse": coarse.state_dict(),
        "fine": fine.state_dict(),
        "near": near, "far": far, "H": H, "W": W,
    }, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    # ==== Stage 6: Novel camera poses ====
    print("\n" + "=" * 60)
    print("Stage 6 – Computing novel camera poses")
    print("=" * 60)

    novel_extri = extri_room1[row_ind].copy()
    novel_intri = intri_room1[row_ind].copy()

    R_s = novel_extri[:, :3, :3]
    t_s = novel_extri[:, :3, 3]
    c_orig = -np.einsum("nij,nj->ni", R_s.transpose(0, 2, 1), t_s)
    c_novel = c_orig + delta_aligned
    t_novel = -np.einsum("nij,nj->ni", R_s, c_novel)
    novel_extri[:, :3, 3] = t_novel

    np.save(os.path.join(args.output_dir, "novel_extrinsics.npy"), novel_extri)
    np.save(os.path.join(args.output_dir, "novel_intrinsics.npy"), novel_intri)
    print(f"  {novel_extri.shape[0]} novel poses saved")
    print(f"  Mean position shift: {np.linalg.norm(delta_aligned, axis=-1).mean():.4f}")

    # ==== Stage 7: Render transferred views ====
    print("\n" + "=" * 60)
    print("Stage 7 – Rendering transferred views")
    print("=" * 60)

    render_dir = os.path.join(args.output_dir, "transferred_views")
    os.makedirs(render_dir, exist_ok=True)

    for i in tqdm(range(novel_extri.shape[0]), desc="  Rendering"):
        img = render_image(
            coarse, fine, novel_extri[i], novel_intri[i],
            H, W, near, far, device=args.device,
        )
        img = np.clip(img, 0.0, 1.0)
        plt.imsave(os.path.join(render_dir, f"transferred_{i:04d}.png"), img)

    # ---- Side-by-side comparison grid ----
    print("  Creating comparison visualisations …")
    cmp_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(cmp_dir, exist_ok=True)

    room2_images = None
    if os.path.isdir(args.images_room2):
        room2_images = load_images_for_nerf(args.images_room2)

    n_cols = 3 if room2_images is not None else 2
    for i in range(novel_extri.shape[0]):
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        axes[0].imshow(images[row_ind[i]])
        axes[0].set_title(f"Room 1 – frame {row_ind[i]}")
        axes[0].axis("off")

        rendered = plt.imread(os.path.join(render_dir, f"transferred_{i:04d}.png"))
        axes[1].imshow(rendered)
        axes[1].set_title(f"Transferred view {i}")
        axes[1].axis("off")

        if room2_images is not None and col_ind[i] < len(room2_images):
            axes[2].imshow(room2_images[col_ind[i]])
            axes[2].set_title(f"Room 2 ref – frame {col_ind[i]}")
            axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(cmp_dir, f"compare_{i:04d}.png"), dpi=150)
        plt.close()

    print(f"\nDone!  Results in {args.output_dir}/")
    print(f"  transferred_views/  – rendered images")
    print(f"  comparisons/        – side-by-side grids")
    print(f"  nerf_checkpoint.pt  – trained model")


if __name__ == "__main__":
    main()
