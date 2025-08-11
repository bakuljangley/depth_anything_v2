import os
import argparse
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib
from vbr_dataset import vbrInterpolatedDataset
from depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm  # Import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Batch depth map estimation")
    parser.add_argument("--vbr_scene", type=str, required=True, help="Name of the scene.")
    parser.add_argument("--vbr_root", type=str, required=True, help="Root directory to dataset.")
    parser.add_argument("--pairs_file", type=str, required=True, help="Path to the pairs CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for depth maps")
    parser.add_argument("--encoder", type=str, choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to DepthAnythingV2 weights file")
    parser.add_argument("--grayscale", action="store_true", help="Save depth maps in grayscale")
    parser.add_argument("--max_depth", type=float, default=100.0, help="(Optional) Max depth for model")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder], max_depth=args.max_depth)
    depth_anything.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    depth_anything = depth_anything.to(device).eval()

    # Load your dataset
    vbr_scene = vbrInterpolatedDataset(args.vbr_root, args.vbr_scene)

    # Read pairs from pairs file: list of (idx1, idx2)
    pairs = []
    with open(args.pairs_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((int(parts[0]), int(parts[1])))

    # Keep track of image paths already processed to avoid duplicates
    processed_images = set()

    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    # Iterate over pairs and run inference on unique images only
    for idx1, idx2 in tqdm(pairs, desc="Processing pairs"):  # Wrap pairs with tqdm
        for idx in (idx1, idx2):
            try:
                img_path = vbr_scene[idx]['image']  # get image filename/path from dataset
            except Exception as e:
                print(f"Warning: cannot get image for index {idx}: {e}")
                continue

            if img_path in processed_images:
                continue

            processed_images.add(img_path)

            raw_image = cv2.imread(img_path)
            if raw_image is None:
                print(f"Warning: Could not load image {img_path}, skipping.")
                continue

            # Run depth inference; add input_size if your infer_image requires it (e.g., 518)
            depth = depth_anything.infer_image(raw_image) 
            basename = os.path.splitext(os.path.basename(img_path))[0]
            raw_depth_out_path = os.path.join(args.output_dir, basename + ".npy")
            np.save(raw_depth_out_path, depth)
            # print(f"Saved raw depth map: {raw_depth_out_path}")
            
            # # Normalize depth to 0-255 uint8 for visualization
            # depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth_norm = depth_norm.astype(np.uint8)

            # if args.grayscale:
            #     out_img = np.repeat(depth_norm[:, :, np.newaxis], 3, axis=-1)
            # else:
            #     out_img = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR for OpenCV

            # basename = os.path.splitext(os.path.basename(img_path))[0]
            # out_path = os.path.join(args.output_dir, basename + "_depth.png")

            # cv2.imwrite(out_path, out_img)
            # print(f"Saved depth map: {out_path}")


if __name__ == "__main__":
    main()