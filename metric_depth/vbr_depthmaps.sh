#!/bin/bash

# --- User Configurable Paths ---
SCENE_NAME="ciampino_train0"
VBR_ROOT="/datasets/vbr_slam/"
PAIRS_PATH="/home/bjangley/VPR/mast3r-v2/pairs_finetuning/ciampino_train0/all_pairs.txt"
OUTPUT_DIR="/home/bjangley/VPR/depthanything_vbr/depth_maps"

# --- Model and Inference Settings ---
ENCODER="vitl"
CHECKPOINT="/home/bjangley/VPR/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"
GRAYSCALE_FLAG="--grayscale"  # set to "--grayscale" if you want grayscale output

MAX_DEPTH=100.0

# --- Device config ---
export CUDA_VISIBLE_DEVICES=4

# --- Launch Depth Anything Inference ---

python create_vbr_depthmaps.py \
    --vbr_scene "$SCENE_NAME" \
    --vbr_root "$VBR_ROOT" \
    --pairs_file "$PAIRS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --encoder "$ENCODER" \
    --ckpt "$CHECKPOINT" \
    --max_depth 100.0 \
    $GRAYSCALE_FLAG
