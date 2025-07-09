import os
import glob
import numpy as np
from tqdm import tqdm
import webdataset as wds
from typing import Union, Tuple

import torch
from torchvision.transforms.functional import InterpolationMode

from transform import image_transform

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

    tf = image_transform(
        image_size=image_size,
        is_train=False,
        mean=None,  # disables normalization
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
        for img_tensor in batch:
            try:
                if isinstance(img_tensor, torch.Tensor):
                    img_np = img_tensor.permute(1, 2, 0).numpy()  # CHW to HWC
                else:
                    continue  # skip non-tensors

                h, w, c = img_np.shape
                n_pixels += h * w

                channel_sum += img_np.sum(axis=(0, 1))
                channel_squared_sum += (img_np ** 2).sum(axis=(0, 1))

                processed += 1
                if processed >= max_images:
                    break
            except Exception as e:
                print(f"Skipping image due to error: {e}")
        if processed >= max_images:
            break

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
