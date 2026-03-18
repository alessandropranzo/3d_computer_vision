#!/bin/bash
set -e

BASE_DIR="/users/eleves-b/2024/alessandro.pranzo/3d_cv_project/3d_computer_vision"
PYTHON_ENV="/users/eleves-b/2024/alessandro.pranzo/.pyenv/versions/gen_ai_env/bin/python"

cd $BASE_DIR

echo "=========================================="
echo "          EXPERIMENTS ROOM 1-2            "
echo "=========================================="

# Room 1-2 NeRF checkpoints
NERF_CKPT_R1="results/models/nerf/room1/nerf_checkpoint.pt"

# Exp 1: ICP + NeRF (room 1-2)
echo "--- Running Exp 1: ICP + NeRF (Room 1-2) ---"
$PYTHON_ENV scripts/nerf_render.py \
    --transfer_dir results/trajectories/room1_2/icp \
    --room1_dir results/vggt_predictions/room1 \
    --images_room1 data/images/room1 \
    --images_room2 data/images/room2 \
    --checkpoint $NERF_CKPT_R1 \
    --output_dir results/experiments/room1_2/exp1_icp_nerf

# Exp 2: VGGT + NeRF (room 1-2)
echo "--- Running Exp 2: VGGT Merged + NeRF (Room 1-2) ---"
$PYTHON_ENV scripts/nerf_render.py \
    --transfer_dir results/trajectories/room1_2/vggt \
    --room1_dir results/vggt_predictions/room1 \
    --images_room1 data/images/room1 \
    --images_room2 data/images/room2 \
    --checkpoint $NERF_CKPT_R1 \
    --output_dir results/experiments/room1_2/exp2_vggt_nerf

# Exp 3: ICP + GS (room 1-2)
echo "--- Running Exp 3: ICP + GS (Room 1-2) ---"
$PYTHON_ENV scripts/gs_render.py \
    --model_path results/models/gs/room1/point_cloud/iteration_3000/point_cloud.ply \
    --transfer_dir results/trajectories/room1_2/icp \
    --images_room1 data/images/room1 \
    --images_room2 data/images/room2 \
    --output_dir results/experiments/room1_2/exp3_icp_gs

# Exp 4: VGGT + GS (room 1-2)
echo "--- Running Exp 4: VGGT Merged + GS (Room 1-2) ---"
$PYTHON_ENV scripts/gs_render.py \
    --model_path results/models/gs/room1/point_cloud/iteration_3000/point_cloud.ply \
    --transfer_dir results/trajectories/room1_2/vggt \
    --images_room1 data/images/room1 \
    --images_room2 data/images/room2 \
    --output_dir results/experiments/room1_2/exp4_vggt_gs


echo "=========================================="
echo "          EXPERIMENTS ROOM 3-4            "
echo "=========================================="

NERF_CKPT_R3="results/models/nerf/room3/nerf_checkpoint.pt"

# Exp 1: ICP + NeRF (room 3-4)
echo "--- Running Exp 1: ICP + NeRF (Room 3-4) ---"
# This will TRAIN the NeRF since checkpoint does not exist, and save it to output_dir
$PYTHON_ENV scripts/nerf_render.py \
    --transfer_dir results/trajectories/room3_4/icp \
    --room1_dir results/vggt_predictions/room3 \
    --images_room1 data/images/room3 \
    --images_room2 data/images/room4 \
    --output_dir results/experiments/room3_4/exp1_icp_nerf

# Move the newly trained checkpoint to models
mv results/experiments/room3_4/exp1_icp_nerf/nerf_checkpoint.pt $NERF_CKPT_R3 || true

# Exp 2: VGGT + NeRF (room 3-4)
echo "--- Running Exp 2: VGGT Merged + NeRF (Room 3-4) ---"
$PYTHON_ENV scripts/nerf_render.py \
    --transfer_dir results/trajectories/room3_4/vggt \
    --room1_dir results/vggt_predictions/room3 \
    --images_room1 data/images/room3 \
    --images_room2 data/images/room4 \
    --checkpoint $NERF_CKPT_R3 \
    --output_dir results/experiments/room3_4/exp2_vggt_nerf

# Exp 3: ICP + GS (room 3-4)
echo "--- Running Exp 3: ICP + GS (Room 3-4) ---"
$PYTHON_ENV scripts/gs_render.py \
    --model_path results/models/gs/room3/point_cloud/iteration_3000/point_cloud.ply \
    --transfer_dir results/trajectories/room3_4/icp \
    --images_room1 data/images/room3 \
    --images_room2 data/images/room4 \
    --output_dir results/experiments/room3_4/exp3_icp_gs

# Exp 4: VGGT + GS (room 3-4)
echo "--- Running Exp 4: VGGT Merged + GS (Room 3-4) ---"
$PYTHON_ENV scripts/gs_render.py \
    --model_path results/models/gs/room3/point_cloud/iteration_3000/point_cloud.ply \
    --transfer_dir results/trajectories/room3_4/vggt \
    --images_room1 data/images/room3 \
    --images_room2 data/images/room4 \
    --output_dir results/experiments/room3_4/exp4_vggt_gs

echo "All 8 experiments completed!"
