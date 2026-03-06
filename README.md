# 3D Room Reconstruction with COLMAP-Free 3D Gaussian Splatting

Reconstruct your room in 3D from an iPhone video using [CF-3DGS](https://github.com/NVlabs/CF-3DGS)
(COLMAP-Free 3D Gaussian Splatting, CVPR 2024). No COLMAP pre-processing required -- the method
jointly estimates camera poses and grows 3D Gaussians from a video sequence.

## Overview

```
iPhone Video  -->  Frame Extraction (Mac)  -->  Training (GPU Server)  -->  Visualization (Mac)
                   extract_frames.py             CF-3DGS                    visualize_trajectory.py
                                                                            visualize_3dgs.py
```

## Requirements

**Local (macOS)** -- for frame extraction and visualization:

```bash
pip install -r requirements_local.txt
```

**Remote (Linux + NVIDIA GPU)** -- for CF-3DGS training:

```bash
# Automated setup:
scp setup_remote.sh user@server:~/
ssh user@server
chmod +x setup_remote.sh && ./setup_remote.sh
```

Or manually follow the steps in `setup_remote.sh`.

## Step 1: Record Your Room

Use the default iPhone Camera app with these guidelines:

- **Resolution**: 1080p (Settings > Camera > Record Video > 1080p at 30fps)
- **Orientation**: Hold the phone in **landscape** mode throughout
- **Movement**: Walk slowly and smoothly. Avoid sudden turns or jerky motions
- **Overlap**: Move slowly enough that consecutive frames share ~80% of the scene
- **Lighting**: Ensure even, consistent lighting. Avoid backlighting from windows
- **Path**: Walk in a continuous path around the room. Starting and ending at the
  same spot (a loop) works well
- **Duration**: 30--60 seconds is ideal (produces 60--150 usable frames at 2 FPS)
- **Avoid**: Reflective surfaces (mirrors, glass), moving objects (people, pets)

Transfer the video to your Mac via AirDrop or cable.

## Step 2: Extract Frames

```bash
# Basic: extract at 2 FPS from a 30fps video
python extract_frames.py --video ~/path/to/room_video.MOV --fps 2

# With resize (recommended for faster training):
python extract_frames.py --video ~/path/to/room_video.MOV --fps 2 --resize 960

# Full options:
python extract_frames.py --video ~/path/to/room_video.MOV \
    --fps 3 \
    --resize 960 \
    --max-frames 200 \
    --output data/my_room/images
```

This saves numbered JPEG frames into `data/my_room/images/`.

## Step 3: Transfer Frames to GPU Server

```bash
scp -r data/my_room user@server:/path/to/CF-3DGS/data/
```

## Step 4: Train on GPU Server

SSH into the server and run:

```bash
conda activate cf3dgs
cd CF-3DGS

python run_cf3dgs.py -s ./data/my_room/ \
                     --mode train \
                     --data_type custom
```

Training takes ~30--60 minutes depending on the number of frames and GPU.

Output is saved to `./output/progressive/my_room/`:
- `chkpnt/ep00_init.pth` -- Gaussian model checkpoint
- `pose/ep00_init.pth` -- estimated camera poses
- `train/` -- per-frame training renders
- `eval/` -- evaluation renders

## Step 5: Transfer Results Back

```bash
scp -r user@server:/path/to/CF-3DGS/output/progressive/my_room/ output/my_room/
```

## Step 6: Visualize

### Camera Trajectory

```bash
# Static plot saved to PNG
python visualize_trajectory.py --pose output/my_room/pose/ep00_init.pth

# Interactive 3D viewer (Open3D)
python visualize_trajectory.py --pose output/my_room/pose/ep00_init.pth --interactive

# Save to custom path
python visualize_trajectory.py --pose output/my_room/pose/ep00_init.pth --save my_trajectory.png
```

### 3D Scene

```bash
# From checkpoint (recommended)
python visualize_3dgs.py --checkpoint output/my_room/chkpnt/ep00_init.pth

# With camera trajectory overlay
python visualize_3dgs.py --checkpoint output/my_room/chkpnt/ep00_init.pth \
                         --pose output/my_room/pose/ep00_init.pth

# Export a simplified PLY for external viewers
python visualize_3dgs.py --checkpoint output/my_room/chkpnt/ep00_init.pth \
                         --export my_room_scene.ply

# Adjust opacity filter and point count
python visualize_3dgs.py --checkpoint output/my_room/chkpnt/ep00_init.pth \
                         --opacity-threshold 0.1 \
                         --max-points 300000
```

The exported PLY can be opened in [MeshLab](https://www.meshlab.net/),
[CloudCompare](https://www.cloudcompare.org/), or uploaded to
[SuperSplat](https://playcanvas.com/supersplat/editor) for web-based viewing.

## Project Structure

```
3d_computer_vision/
    README.md                   # This file
    requirements_local.txt      # macOS dependencies
    extract_frames.py           # Video to frames
    visualize_trajectory.py     # Camera trajectory viewer
    visualize_3dgs.py           # 3D scene viewer
    setup_remote.sh             # GPU server setup script
    data/
        my_room/
            images/             # Extracted JPEG frames
    output/                     # Results from training
        my_room/
            chkpnt/
            pose/
            train/
            eval/
```

## References

- [COLMAP-Free 3D Gaussian Splatting](https://oasisyang.github.io/colmap-free-3dgs/) (Fu et al., CVPR 2024)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al., SIGGRAPH 2023)

```bibtex
@InProceedings{Fu_2024_CVPR,
    author    = {Fu, Yang and Liu, Sifei and Kulkarni, Amey and Kautz, Jan and Efros, Alexei A. and Wang, Xiaolong},
    title     = {COLMAP-Free 3D Gaussian Splatting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {20796-20805}
}
```
