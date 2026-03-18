#!/usr/bin/env python3
"""
NeRF Train & Render
===================
Train a coarse+fine NeRF on room-1 images, then render novel views
from transferred camera poses (produced by trajectory_transfer.py).

Stages:
  1. Train NeRF on room-1 images + standalone camera poses.
  2. Render transferred views from novel camera poses.
  3. Create side-by-side comparison grids.

Usage:
    python nerf_render.py \\
        --room1_dir predictions_visuals/room1 \\
        --images_room1 data/images/room1 \\
        --images_room2 data/images/room2 \\
        --transfer_dir transfer_results \\
        --output_dir nerf_results

    # Render-only (skip training, load existing checkpoint):
    python nerf_render.py \\
        --checkpoint nerf_results/nerf_checkpoint.pt \\
        --transfer_dir transfer_results \\
        --output_dir nerf_results
"""

import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# Image Loading
# ============================================================

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


# ============================================================
# Ray Helpers
# ============================================================

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


# ============================================================
# Sampling
# ============================================================

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
    """Inverse-CDF sampling from a piecewise-constant PDF."""
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


# ============================================================
# Volume Rendering
# ============================================================

def volume_render(rgb, sigma, t_vals, rays_d):
    """Classic NeRF alpha-compositing.

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
    """Train coarse + fine NeRF.

    images:     (N, H, W, 3) float32 [0,1]
    extrinsics: (N, 3, 4)
    intrinsics: (N, 3, 3)
    """
    N_img, H, W, _ = images.shape
    near, far = estimate_scene_bounds(extrinsics)
    print(f"  Scene bounds: near={near:.4f}, far={far:.4f}")

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

        # coarse pass
        pts_c, t_c = stratified_sample(ro, rd, near, far, N_coarse, perturb=True)
        d_c = rd[:, None, :].expand_as(pts_c)
        rgb_c, sig_c = coarse(pts_c.reshape(-1, 3), d_c.reshape(-1, 3))
        rgb_c = rgb_c.view(batch_size, N_coarse, 3)
        sig_c = sig_c.view(batch_size, N_coarse, 1)
        rend_c, w_c, _, acc_c = volume_render(rgb_c, sig_c, t_c, rd)

        # fine pass (hierarchical)
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

        # losses
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NeRF Train & Render",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--room1_dir", default="results/vggt_predictions/room1",
                        help="VGGT predictions for room 1 (extrinsics, intrinsics)")
    parser.add_argument("--images_room1", default="data/images/room1")
    parser.add_argument("--images_room2", default="data/images/room2")
    parser.add_argument("--transfer_dir", default="results/trajectory_transfers/base",
                        help="Output of trajectory_transfer.py")
    parser.add_argument("--output_dir", default="results/nerf_renders/base")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to existing checkpoint (skip training)")
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load transfer results ──────────────────────────────────────────
    novel_extri = np.load(os.path.join(args.transfer_dir, "novel_extrinsics.npy"))
    novel_intri = np.load(os.path.join(args.transfer_dir, "novel_intrinsics.npy"))
    row_ind = np.load(os.path.join(args.transfer_dir, "match_row_ind.npy"))
    col_ind = np.load(os.path.join(args.transfer_dir, "match_col_ind.npy"))
    print(f"Loaded {novel_extri.shape[0]} novel poses from {args.transfer_dir}/")

    # ── Load room-1 data ───────────────────────────────────────────────
    extri_room1 = np.load(os.path.join(args.room1_dir, "extrinsics.npy"))
    intri_room1 = np.load(os.path.join(args.room1_dir, "intrinsics.npy"))

    images = load_images_for_nerf(args.images_room1)
    H, W = images.shape[1], images.shape[2]
    print(f"Room 1 images: {images.shape[0]} × {H}×{W}")

    # ── Train or load NeRF ─────────────────────────────────────────────
    ckpt_path = args.checkpoint or os.path.join(args.output_dir, "nerf_checkpoint.pt")

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        coarse = NeRFMLP().to(args.device)
        fine = NeRFMLP().to(args.device)
        coarse.load_state_dict(ckpt["coarse"])
        fine.load_state_dict(ckpt["fine"])
        near, far = ckpt["near"], ckpt["far"]
        H, W = ckpt["H"], ckpt["W"]
        coarse.eval()
        fine.eval()
    else:
        print("\n" + "=" * 60)
        print("Training NeRF on room 1")
        print("=" * 60)
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

    # ── Render transferred views ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Rendering transferred views")
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

    # ── Comparison grids ───────────────────────────────────────────────
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
