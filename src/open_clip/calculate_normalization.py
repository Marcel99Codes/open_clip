import os
import glob
import numpy as np
from tqdm import tqdm
import webdataset as wds
from typing import Union, Tuple

import torch
from torchvision.transforms.functional import InterpolationMode

from open_clip.transform import image_transform


def compute_streaming_mean_std(
    shards_dir: str,
    image_size: Union[int, Tuple[int, int]] = 224,
    color_space: str = "rgb",
    max_images: int = 10000,
    batch_size: int = 64,
):
    # Distributed setup (optional)
    node_rank = int(os.environ.get("RANK", 0))
    num_nodes = int(os.environ.get("WORLD_SIZE", 1))

    # Collect shards for this node
    all_shards = sorted(glob.glob(os.path.join(shards_dir, "*.tar")))
    if num_nodes > 1:
        shards_for_node = [shard for i, shard in enumerate(all_shards) if i % num_nodes == node_rank]
    else:
        shards_for_node = all_shards

    # Transform without normalization
    tf = image_transform(
        image_size=image_size,
        is_train=False,
        mean=None,
        std=None,
        color_space=color_space,
        resize_mode="shortest",
        interpolation="bicubic",
        aug_cfg=None
    )

    dataset = (
        wds.WebDataset(shards_for_node)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg")
        .map_tuple(tf)
        .batched(batch_size)
    )

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    processed = 0

    for batch in tqdm(dataset, desc="Computing mean/std"):
        try:
            if not isinstance(batch, torch.Tensor) and isinstance(batch, (tuple, list)):
                batch = batch[0]

            if batch.ndim != 4 or batch.shape[1] != 3:
                print(f"Skipping batch with unexpected shape: {batch.shape}")
                continue

            # batch shape: [B, 3, H, W] -> [B, H, W, 3]
            batch_np = batch.permute(0, 2, 3, 1).numpy()
            h, w = batch_np.shape[1:3]
            pixels_in_batch = batch_np.shape[0] * h * w

            channel_sum += batch_np.sum(axis=(0, 1, 2))
            channel_squared_sum += (batch_np ** 2).sum(axis=(0, 1, 2))

            n_pixels += pixels_in_batch
            processed += batch_np.shape[0]

            if processed >= max_images:
                break

        except Exception as e:
            print(f"Skipping batch due to error: {e}")
            continue

    if processed == 0 or n_pixels == 0:
        raise RuntimeError("No valid images were processed.")

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_squared_sum / n_pixels - mean ** 2)

    print(f"\n[RESULT] Dataset mean: {mean.tolist()}")
    print(f"[RESULT] Dataset std:  {std.tolist()}")
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_dir", type=str, required=True, help="Directory containing .tar WebDataset shards")
    parser.add_argument("--image_size", type=int, default=224, help="Image size used for transforms")
    parser.add_argument("--max_images", type=int, default=10000, help="Max images to process")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--color_space", type=str, default="rgb", choices=["rgb", "hsv", "ycbcr", "lab"])
    args = parser.parse_args()

    compute_streaming_mean_std(
        shards_dir=args.shards_dir,
        image_size=args.image_size,
        color_space=args.color_space,
        max_images=args.max_images,
        batch_size=args.batch_size
    )
