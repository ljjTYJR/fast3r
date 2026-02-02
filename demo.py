#!/usr/bin/env python3
"""Fast3R pose estimation for multi-camera sequences."""

import torch
import os
import glob
import numpy as np
import argparse
import time
import json
from scipy.spatial.transform import Rotation

from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.utils.checkpoint_utils import load_model
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule


def write_tum_poses(poses, indices, path, source_dir, sampling_interval):
    """Write poses to file in TUM format with frame indices instead of timestamps."""
    with open(path, 'w') as f:
        f.write("# TUM format: index tx ty tz qx qy qz qw\n")
        f.write("# Camera-to-world transformation\n")
        f.write(f"# Data source: {source_dir}\n")
        f.write(f"# Sampling interval: {sampling_interval}\n")
        f.write(f"# Note: index represents original frame index (considering sampling interval)\n")

        for idx, pose in zip(indices, poses):
            t = pose[:3, 3]
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()  # [x, y, z, w]
            f.write(f"{idx} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")


def save_depth_maps(preds, indices, output_dir, cam_name):
    """Extract and save depth maps from 3D point clouds.

    Args:
        preds: List of prediction dicts containing 'pts3d_in_other_view' (H, W, 3) point clouds
        indices: List of frame indices
        output_dir: Directory to save depth maps
        cam_name: Camera name for file naming
    """
    depth_dir = os.path.join(output_dir, f"{cam_name}_depth")
    os.makedirs(depth_dir, exist_ok=True)

    depth_stats = []
    for idx, pred in zip(indices, preds):
        # Extract 3D points: shape is (H, W, 3) where last dim is (x, y, z)
        pts3d = pred['pts3d_in_other_view']
        if isinstance(pts3d, torch.Tensor):
            pts3d = pts3d.cpu().numpy()

        # Remove batch dimension if present: (1, H, W, 3) -> (H, W, 3)
        if pts3d.ndim == 4 and pts3d.shape[0] == 1:
            pts3d = pts3d[0]

        # Extract depth (z-coordinate)
        depth = pts3d[:, :, 2]

        # Save as numpy array (.npy) for precise values
        depth_file = os.path.join(depth_dir, f"{idx:06d}.npy")
        np.save(depth_file, depth.astype(np.float32))

        # Collect depth statistics
        depth_valid = depth[np.isfinite(depth)]
        if len(depth_valid) > 0:
            depth_stats.append({
                'index': int(idx), 'min': float(depth_valid.min()), 'max': float(depth_valid.max()),
                'mean': float(depth_valid.mean()), 'median': float(np.median(depth_valid))
            })

    # Save depth statistics
    stats_file = os.path.join(output_dir, f"{cam_name}_depth_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(depth_stats, f, indent=2)

    return len(preds)


def process_cameras(cam_dirs, cam_names, model, device, args, output_dir, model_load_time):
    """Process multiple cameras together for better pose estimation."""
    print(f"\n{'='*60}\nProcessing {len(cam_names)} cameras together\n{'='*60}")

    all_files = []
    all_indices = []
    cam_file_counts = []

    # Collect files from all cameras
    for cam_dir, cam_name in zip(cam_dirs, cam_names):
        files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
        if not files:
            files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
        if not files:
            print(f"Warning: No images in {cam_dir}")
            cam_file_counts.append(0)
            continue

        # Calculate indices before sampling (to preserve original frame numbers)
        indices = list(range(0, len(files), args.sampling_interval))
        files = files[::args.sampling_interval]
        cam_file_counts.append(len(files))

        all_files.extend(files)
        all_indices.append(indices)
        print(f"{cam_name}: {len(files)} images")

    if not all_files:
        print("No images found in any camera")
        return {}

    total_images = len(all_files)
    print(f"\nTotal: {total_images} images from {len(cam_names)} cameras")

    t0 = time.time()

    # Load all images
    print("Loading images...")
    t1 = time.time()
    images = load_images(all_files, size=args.image_size, verbose=False,
                        rotate_clockwise_90=args.rotate_90, crop_to_landscape=args.crop_landscape)
    load_time = time.time() - t1
    print(f"  Load: {load_time:.2f}s")

    # Run inference on all images together
    print("Running inference...")
    t1 = time.time()
    with torch.no_grad():
        output, profile = inference(images, model, device, dtype=torch.float32, verbose=False, profiling=True)
    inf_time = profile.get('total_time', 0)
    print(f"  Inference: {inf_time:.2f}s")

    # Estimate poses
    print("Estimating poses...")
    t1 = time.time()
    poses_list, focals_list = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output['preds'], niter_PnP=100, focal_length_estimation_method='first_view_from_global_head'
    )
    pose_time = time.time() - t1
    print(f"  Pose estimation: {pose_time:.2f}s")

    # Get poses and focal from first batch
    poses = poses_list[0]
    focal = focals_list[0][0] if isinstance(focals_list[0], list) else focals_list[0]

    # Extract predictions before cleanup
    preds = output['preds']

    # Cleanup
    del images, output
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    total = time.time() - t0

    # Split poses and predictions by camera, then save
    results = {}
    start_idx = 0
    for cam_name, cam_dir, count, indices in zip(cam_names, cam_dirs, cam_file_counts, all_indices):
        if count == 0:
            continue

        cam_poses = poses[start_idx:start_idx + count]
        cam_preds = preds[start_idx:start_idx + count]
        start_idx += count

        # Save poses
        pose_file = os.path.join(output_dir, f"{cam_name}_poses.txt")
        write_tum_poses(cam_poses, indices, pose_file, cam_dir, args.sampling_interval)

        # Save depth maps
        print(f"Saving depth maps for {cam_name}...")
        num_depth = save_depth_maps(cam_preds, indices, output_dir, cam_name)

        meta = {
            'camera': cam_name, 'num_images': count, 'num_poses': len(cam_poses),
            'num_depth_maps': num_depth, 'image_size': args.image_size,
            'sampling_interval': args.sampling_interval, 'focal_length': float(focal),
            'shared_processing': True
        }

        with open(os.path.join(output_dir, f"{cam_name}_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        results[cam_name] = meta
        print(f"{cam_name}: {len(cam_poses)} poses saved")

    print(f"\nTotal time: {total:.1f}s ({total/total_images:.2f}s/img)")
    print(f"Focal length: {focal:.1f}px")

    # Save shared timing info
    timing = {
        'total_images': total_images, 'num_cameras': len(cam_names),
        'model_load_time': model_load_time, 'load_time': load_time,
        'inference_time': inf_time, 'pose_time': pose_time,
        'total_time': total, 'avg_per_image': total / total_images
    }
    with open(os.path.join(output_dir, "timing.json"), 'w') as f:
        json.dump(timing, f, indent=2)

    return results


def main():
    p = argparse.ArgumentParser(description="Fast3R pose estimation")
    p.add_argument("--data_dir", default="/media/shuo/T7/robolab/scripts/processed_clips/test/robodog_01")
    p.add_argument("--output_dir", default=None, help="Output dir (default: {data_dir}/fast3r_poses)")
    p.add_argument("--cameras", nargs='+', default=["cam01", "cam02"], help="Camera subdirs to process")
    p.add_argument("--checkpoint", default="jedyang97/Fast3R_ViT_Large_512", help="Model checkpoint path/ID")
    p.add_argument("--lightning", action="store_true", help="Use Lightning checkpoint loader")
    p.add_argument("--sampling_interval", type=int, default=2, help="Use every Nth frame")
    p.add_argument("--image_size", type=int, default=512, choices=[224, 512])
    p.add_argument("--rotate_90", action="store_true", help="Rotate images 90° clockwise")
    p.add_argument("--crop_landscape", action="store_true", help="Crop to landscape")
    args = p.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(args.data_dir, "fast3r_poses")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\n" + "="*60)
    print("Loading model...")
    model_load_start = time.time()
    model, _ = load_model(args.checkpoint, device=device, is_lightning_checkpoint=args.lightning)
    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.1f}s")

    # Process all cameras together
    cam_dirs = [os.path.join(args.data_dir, cam) for cam in args.cameras]
    valid_cams = []
    valid_dirs = []
    for cam, cam_dir in zip(args.cameras, cam_dirs):
        if os.path.exists(cam_dir):
            valid_cams.append(cam)
            valid_dirs.append(cam_dir)
        else:
            print(f"Skipping {cam}: directory not found")

    if not valid_cams:
        print("No valid camera directories found")
        return

    results = process_cameras(valid_dirs, valid_cams, model, device, args, args.output_dir, model_load_time)

    # Save summary
    summary = {
        'data_dir': args.data_dir, 'output_dir': args.output_dir,
        'checkpoint': args.checkpoint, 'device': str(device),
        'cameras': list(results.keys()), 'results': results
    }
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(results)} cameras: {', '.join(results.keys())}")
    print(f"Results in: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()