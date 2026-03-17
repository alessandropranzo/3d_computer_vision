# 3D Computer Vision: Room Reconstruction & Trajectory Transfer Pipeline

## Introduction / Problem Definition

Given two videos of similar environments, the goal is to transfer the camera trajectory from one environment to the other.

## Pipeline Overview

```mermaid
flowchart TD
    subgraph Step 1: 3D Feature Extraction
        Video1[Video 1]
        Video2[Video 2]
        VGGT1[VGGT Features 1\n(Poses, Point Cloud)]
        VGGT2[VGGT Features 2\n(Poses, Point Cloud)]
    end
    
    subgraph Step 2: 3D Environment Alignment
        Alignment{Alignment Method\n1. VGGT Concatenated\n2. ICP}
        TransferredTraj[Transferred Trajectory]
    end
    
    subgraph Step 3: Scene Representation of Target
        SceneRep{Scene Representation\n1. NeRF\n2. Gaussian Splatting}
    end
    
    subgraph Step 4: Rendering
        Renderer[Render First Environment]
        NovelViews[Novel Views]
    end

    %% Data Flow
    Video1 --> VGGT1
    Video2 --> VGGT2
    
    VGGT1 --> Alignment
    VGGT2 --> Alignment
    Alignment --> TransferredTraj
    
    Video1 --> SceneRep
    
    SceneRep --> Renderer
    TransferredTraj --> Renderer
    Renderer --> NovelViews
```

## Directory Structure

- `notebooks/`: Jupyter notebooks for experimentation and prototyping.
- `results/` or `output/`: Folders for pipeline outputs (predictions, renderings, transfers).
- `data/`: Source videos and extracted image frames.

## Requirements

To install the necessary local dependencies (requires Python 3.8+):

```bash
pip install -r requirements.txt
```

For advanced GPU-based components, a remote Linux machine with an NVIDIA GPU is recommended.

## Step-by-Step Pipeline

### Step 1: 3D Feature Extraction

Starting from the two input videos, we independently extract 3D features for each environment using VGGT. First, extract the frames from the input videos:

```bash
# Basic extraction at 2 FPS
python extract_frames.py --video path/to/video.MOV --output data/images/room1 --fps 2
```

Then, extract 3D point clouds and camera poses (extrinsics/intrinsics) from the image sequences using the VGGT model:

```bash
python extract_visualizations.py \
    --path1 data/images/room1/ \
    --path2 data/images/room2/ \
    --out_dir data/vggt_predictions/
```

### Step 2: 3D Environment Alignment

Once the 3D features are obtained, we need to align the two environments in the same 3D space.

We consider two possible methods for this alignment:
- **VGGT with concatenated images** (Procrustes-based alignment approach)
- **ICP (Iterative Closest Point)** between the two reconstructed rooms

The output of this step is a transferred trajectory, represented as a list of camera positions from the second environment mapped into the coordinate system of the first environment.

```bash
# Using ICP
python trajectory_transfer_icp.py \
    --room1_dir data/vggt_predictions/room1 \
    --room2_dir data/vggt_predictions/room2 \
    --output_dir transfer_results_icp

# Or using the VGGT concatenated Procrustes approach
python trajectory_transfer.py \
    --merged_dir data/vggt_predictions/merged \
    --room1_dir data/vggt_predictions/room1 \
    --images_room1 data/images/room1 \
    --output_dir transfer_results
```

### Step 3: Scene Representation of the Target Environment

Next, we want to render views in the first environment from these transferred camera positions. 

To do this, we train a scene representation model on the frames of the first environment. We consider two possible approaches:
- **NeRF (Neural Radiance Fields)**
- **Gaussian Splatting (GS)**

**For NeRF:**
```bash
python nerf_render.py \
    --transfer_dir transfer_results \
    --room1_dir data/vggt_predictions/room1 \
    --images_room1 data/images/room1 \
    --output_dir nerf_results
```

**For Gaussian Splatting:**
First, prepare the data into COLMAP format:
```bash
python prepare_gs_data.py \
    --room1_dir data/vggt_predictions/room1 \
    --images_dir data/images/room1 \
    --output_dir data/colmap_room1
```
Then, train using the official INRIA 3DGS repository.

### Step 4: Rendering

Finally, we render the first environment using the transferred camera trajectory.

**Rendering with Gaussian Splatting:**
```bash
python gs_render.py \
    --model_path output/gs_room1/point_cloud/iteration_3000/point_cloud.ply \
    --transfer_dir transfer_results \
    --images_room1 data/images/room1 \
    --images_room2 data/images/room2 \
    --output_dir gs_results
```

*(You can also generate interactive 3D HTML visualizations to compare trajectories and point clouds before and after the transfer!)*
```bash
python visualize_scenes.py \
    --merged_dir data/vggt_predictions/merged \
    --room1_dir data/vggt_predictions/room1 \
    --room2_dir data/vggt_predictions/room2 \
    --images_room1 data/images/room1 \
    --transfer_dir transfer_results \
    --output_dir visualizations
```
Outputs are saved as HTML files. Open them in any web browser to explore the 3D scene.
