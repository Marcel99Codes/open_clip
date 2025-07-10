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
    image_size: Union[int, Tuple[int, int]] = 224,
    color_space: str = "rgb",
    max_images: int = 10000,
    batch_size: int = 64,
    save_path: str = None,
):
    # Distributed setup
    node_rank = int(os.environ.get("RANK", 0))
    num_nodes = int(os.environ.get("WORLD_SIZE", 1))

    all_shards = sorted(glob.glob(os.path.join(shards_dir, "*.tar")))
    if num_nodes > 1:
        shards_for_node = [shard for i, shard in enumerate(all_shards) if i % num_nodes == node_rank]
    else:
        shards_for_node = all_shards

    tf = image_transform(
        image_size=image_size,
        is_train=False,
        mean=None,  # No normalization
        std=None,
        color_space=color_space,
        resize_mode="shortest",
        interpolation="bicubic",
        aug_cfg=None,
    )

    dataset = (
        wds.WebDataset(shards_for_node)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg")
        .map_tuple(tf)
        .batched(batch_size)
    )

    channel_sum = None
    channel_squared_sum = None
    n_pixels = 0
    processed = 0

    for batch in tqdm(dataset, desc="Computing mean/std"):
        if not isinstance(batch, (list, tuple)):
            continue
        for img_tensor in batch:
            try:
                if isinstance(img_tensor, torch.Tensor):
                    if img_tensor.ndim == 3:
                        pass  # Single image: [C, H, W]
                    elif img_tensor.ndim == 4:
                        raise ValueError(f"Unexpected batch image shape: {img_tensor.shape}")
                    else:
                        continue

                    img_np = img_tensor.permute(1, 2, 0).numpy()  # [H, W, C]

                    if channel_sum is None:
                        channel_sum = np.zeros(img_np.shape[2], dtype=np.float64)
                        channel_squared_sum = np.zeros(img_np.shape[2], dtype=np.float64)

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

    if n_pixels == 0:
        raise ValueError("No valid images were processed.")

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_squared_sum / n_pixels - mean ** 2)

    print(f"\n[RESULT] Dataset mean ({color_space}): {mean.tolist()}")
    print(f"[RESULT] Dataset std  ({color_space}): {std.tolist()}")

    if save_path is None:
        save_path = f"./normalization/norm_{color_space}.json"

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
    parser.add_argument("--color_space", type=str, default="rgb", choices=["rgb", "hsv", "ycbcr", "lab"])
    parser.add_argument("--save_path", type=str, default=None, help="Path to save mean/std JSON")

    args = parser.parse_args()

    compute_streaming_mean_std(
        shards_dir=args.shards_dir,
        image_size=args.image_size,
        color_space=args.color_space,
        max_images=args.max_images,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
