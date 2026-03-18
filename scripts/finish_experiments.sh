#!/bin/bash
set -e

BASE_DIR="/users/eleves-b/2024/alessandro.pranzo/3d_cv_project/3d_computer_vision"
PYTHON_ENV="/users/eleves-b/2024/alessandro.pranzo/.pyenv/versions/gen_ai_env/bin/python"
NERF_CKPT_R3="results/models/nerf/room3/nerf_checkpoint.pt"

echo "Waiting for PID 2107340 to finish..."
tail --pid=2107340 -f /dev/null

echo "Moving newly trained NeRF checkpoint to results/models/nerf/room3/"
mv results/experiments/room3_4/exp1_icp_nerf/nerf_checkpoint.pt $NERF_CKPT_R3 || true

echo "Running Exp 2: VGGT Merged + NeRF (Room 3-4)"
$PYTHON_ENV scripts/nerf_render.py \
    --transfer_dir results/trajectories/room3_4/vggt \
    --room1_dir results/vggt_predictions/room3 \
    --images_room1 data/images/room3 \
    --images_room2 data/images/room4 \
    --checkpoint $NERF_CKPT_R3 \
    --output_dir results/experiments/room3_4/exp2_vggt_nerf

echo "Done!"
