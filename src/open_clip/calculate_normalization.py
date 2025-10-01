import os
import glob
import numpy as np
from tqdm import tqdm
import webdataset as wds
from typing import Union, Tuple
import json

import torch
from torchvision.transforms.functional import InterpolationMode
from open_clip.transform import image_transform


def compute_streaming_mean_std(
    shards_dir: str,
    image_size: Union[int, Tuple[int, int]],
    color_space: str,
    max_images: int,
    batch_size: int,
    save_in_file : bool,
    save_path: str,

):
    all_shards = sorted(os.path.join(shards_dir, f) for f in os.listdir(shards_dir) if f.endswith('.tar'))

    transformation = image_transform(
        image_size=image_size,
        is_train=False,
        mean=None,
        std=None,
        color_space=color_space,
        resize_mode="shortest",
        interpolation="bicubic",
        aug_cfg=None,
        no_default_norm=True,
    )

    dataset = (wds.WebDataset(all_shards)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg")
        .map_tuple(transformation)
        .batched(batch_size)
    )

    sum = torch.zeros(3, dtype=torch.float64)
    squared_sum = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    processed = 0

    # Tqdm wraps iterable with a progress bar
    for batch in tqdm(dataset, desc="Computing mean/std"):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        # Expect shape [B, 3, H, W]
        if batch.ndim != 4 or batch.shape[1] != 3:
            print(f"Skipping batch with unexpected shape: {batch.shape}")
            continue

        #batch = batch.to(torch.float32)
        batch = batch.to(torch.float32)

        sum += batch.sum(dim=(0, 2, 3)) #sum per channel
        squared_sum += (batch**2).sum(dim=(0, 2, 3)) #squared sum per channel

        n_pixels += batch.shape[0] * batch.shape[2] * batch.shape[3]
        processed += batch.shape[0]

        if processed >= max_images:
            break

    mean = sum / n_pixels
    std = torch.sqrt((squared_sum - (sum**2) / n_pixels) / (n_pixels - 1))

    print(f"\n[RESULT] Dataset mean ({color_space}): {mean.tolist()}")
    print(f"[RESULT] Dataset std  ({color_space}): {std.tolist()}")

    if save_in_file:
        if save_path is None:
            save_path = f"./normalization/norm_{color_space}.json"
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
            print(f"[INFO] Saved normalization stats to {save_path}")

    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_dir", type=str, required=True, help="Directory with .tar WebDataset shards")
    parser.add_argument("--image_size", type=int, default=224, help="Image size used for transforms")
    parser.add_argument("--max_images", type=int, default=10000, help="Max number of images to process")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--color_space", type=str, default="ycbcr", choices=["rgb", "hsv", "ycbcr", "lab"])
    parser.add_argument("--save_in_file", action="store_true", help="Whether to save the results in a file")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save mean/std JSON")

    args = parser.parse_args()

    compute_streaming_mean_std(
        shards_dir=args.shards_dir,
        image_size=args.image_size,
        color_space=args.color_space,
        max_images=args.max_images,
        batch_size=args.batch_size,
        save_path=args.save_path,
        save_in_file=args.save_in_file
    )
