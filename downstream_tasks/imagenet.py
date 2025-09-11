import argparse
import torch
import sys, os
from datasets import load_dataset
from tqdm import tqdm
import csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import open_clip
from datasets import load_dataset
from tqdm import tqdm
import csv

os.environ["HF_HUB_ENABLE_XET"] = "0"



# -------------------------
# Helper: Fix checkpoint keys
# -------------------------
def clean_state_dict(state_dict):
    """Remove common prefixes like 'module.', 'model.', 'clip.' from checkpoint keys."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("clip."):
            k = k[len("clip."):]
        new_sd[k] = v
    return new_sd


# -------------------------
# Dataset Loader
# -------------------------
def load_data_and_classes(dataset_name, preprocess, class_subset=None):
    if dataset_name == "imagenet1k":
        dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="validation")
        labels = dataset.features["label"].names
        # Use integer indices for class_names
        class_names = list(range(len(labels)))
        key = "label"

    elif dataset_name == "imagenet_a":
        # WebDataset-based ImageNet-A
        dataset = load_dataset("clip-benchmark/wds_imagenet-a", split="test", streaming=True)
        key = "cls"
        # Use integer indices for class_names
        class_names = list(range(200))  # ImageNet-A has 200 classes

    elif dataset_name == "imagenet_o":
        # WebDataset-based ImageNet-R
        dataset = load_dataset("clip-benchmark/wds_imagenet-o", split="test", streaming=True)
        key = "cls"
        # Use integer indices for class_names
        class_names = list(range(200))  # ImageNet-A has 200 classes

    elif dataset_name == "imagenet_c":
        dataset = load_dataset("ang9867/ImageNet-C", split="validation", streaming=True)
        labels = dataset.features["label"].names
        # Use integer indices for class_names
        class_names = list(range(len(labels)))
        key = "label"

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Handle subset filtering
    if class_subset is not None:
        indices = [int(i) for i in class_subset.split(",")]
        class_names = [i for i in indices]  # keep as integers
        valid_set = set(indices)
        dataset = dataset.filter(lambda x: x[key] in valid_set)

        # Remap label → new index (0..len(indices)-1)
        idx_map = {old: new for new, old in enumerate(indices)}
        dataset = dataset.map(lambda x: {key: idx_map[x[key]]})

    return dataset, class_names



# -------------------------
# Evaluation
# -------------------------
def evaluate(dataset, class_names, text_features, batch_size=64, preprocess=None, device="cpu"):
    top1, top5, total = 0, 0, 0
    images, labels = [], []

    for sample in tqdm(dataset, desc="Evaluating", leave=False):
        key = "image" if "image" in sample else "jpg"
        images.append(preprocess(sample[key]))

        key = "label" if "label" in sample else "cls"
        labels.append(sample[key])

        if len(images) == batch_size:
            imgs = torch.stack(images).to(device)
            labs = torch.tensor(labels)

            with torch.no_grad():
                image_features = model.encode_image(imgs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            _, indices = logits.topk(5, dim=-1)

            top1 += (indices[:, 0].cpu() == labs).sum().item()
            for i, l in enumerate(labs):
                if l in indices[i].cpu().numpy():
                    top5 += 1
            total += len(labs)
            images, labels = [], []

    return top1, top5, total


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["imagenet1k", "imagenet_a", "imagenet_c", "imagenet_o"],
                        help="Dataset to evaluate on")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs/checkpoints to evaluate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--class_subset", type=str, default=None,
                        help="Comma-separated list of classes to evaluate, e.g., '0,5,10' or 'goldfish,great white shark'")
    parser.add_argument("--csv_file", type=str, default="evaluation_results.csv",
                        help="CSV file to save results")

    args = parser.parse_args()

    # -------------------------
    # Hardcoded checkpoint directories for the 4 runs
    # -------------------------
    checkpoint_config_small =[
        {
            "path": "/data1/marcel/clip/models_dc_small/rgb/2025_07_15-10_03_33-model_ViT-B-16-lr_0.0005-b_512-j_8-p_amp/checkpoints", 
            "colorspace": "rgb", 
            "pipeline": "single",
            "conv1_patch_size": 16,
            "convb_patch_size": 16,
            "grayscale_only": False
        },
        {
            "path": "/data1/marcel/clip/models_dc_small/ycbcr/2025_07_14-16_39_55-model_ViT-B-16-lr_0.0005-b_512-j_8-p_amp/checkpoints",
            "colorspace": "ycbcr", 
            "pipeline": "dual_c1",
            "conv1_patch_size": 16,
            "convb_patch_size": 64,
            "grayscale_only": False
        },
        {
            "path": "/data1/marcel/clip/models_dc_small/grayscale/2025_07_17-06_47_13-model_ViT-B-16-lr_0.0005-b_512-j_8-p_amp/checkpoints",
            "colorspace": "ycbcr", 
            "pipeline": "dual_c1",
            "conv1_patch_size": 16,
            "convb_patch_size": 16,
            "grayscale_only": True
        },
        {
            "path": "/data1/marcel/clip/models_dc_small/lab/2025_07_16-12_14_53-model_ViT-B-16-lr_0.0005-b_512-j_8-p_amp/checkpoints",
            "colorspace": "lab", 
            "pipeline": "dual_c1",
            "conv1_patch_size": 16,
            "convb_patch_size": 56,
            "grayscale_only": False
        },
    ]

    checkpoint_config_medium =[
        {
            "path": "/data1/marcel/clip/models_dc_medium/rgb/2025_09_07-09_47_10-model_ViT-B-16-lr_5e-05-b_512-j_2-p_amp/checkpoints", 
            "colorspace": "rgb", 
            "pipeline": "single",
            "conv1_patch_size": 16,
            "convb_patch_size": 16,
            "grayscale_only": False
        },
        {
            "path": "/data1/marcel/clip/models_dc_medium/ycbcr/2025_09_07-09_46_58-model_ViT-B-16-lr_5e-05-b_512-j_2-p_amp/checkpoints",
            "colorspace": "ycbcr", 
            "pipeline": "dual_c1",
            "conv1_patch_size": 16,
            "convb_patch_size": 56,
            "grayscale_only": False
        },
        {
            "path": "/data1/marcel/clip/models_dc_medium/grayscale/2025_09_07-09_47_42-model_ViT-B-16-lr_5e-05-b_512-j_2-p_amp/checkpoints",
            "colorspace": "ycbcr", 
            "pipeline": "dual_c1",
            "conv1_patch_size": 16,
            "convb_patch_size": 16,
            "grayscale_only": True
        },
    ]

    # -------------------------
    # Evaluate all runs & checkpoints
    # -------------------------
    results = []

    for run_idx, ckpt_config in enumerate(checkpoint_config_medium):
        run_name = f"run_{run_idx+1}"
        print(f"\nEvaluating model: {run_name}")

        # Device + model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "ViT-B-16"

        open_clip.transformer_args.set_transformer_args(ckpt_config["pipeline"], ckpt_config["conv1_patch_size"], ckpt_config["convb_patch_size"], ckpt_config["grayscale_only"])
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None, colorspace=ckpt_config["colorspace"]
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device)

        # Load dataset (no text features yet)
        dataset, class_names = load_data_and_classes(
            args.dataset, preprocess, args.class_subset
        )

        for epoch in range(1, args.epochs + 1):
            ckpt_path = os.path.join(ckpt_config["path"], f"epoch_{epoch}.pt")
            print(f"  Loading checkpoint: {ckpt_path}")

            print(ckpt_config)

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("ema_state_dict", checkpoint.get("state_dict", checkpoint))
            print(state_dict)
            state_dict = clean_state_dict(state_dict)
            print(state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Epoch {epoch}: missing {len(missing)} keys, unexpected {len(unexpected)} keys")

            model = model.to(device)
            model.eval()

            # Recompute text features for this checkpoint
            text_inputs = tokenizer([f"a photo of a {c}" for c in class_names]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # Evaluate
            top1, top5, total = evaluate(dataset, class_names, text_features,
                                         batch_size=args.batch_size,
                                         preprocess=preprocess,
                                         device=device)
            acc1, acc5 = 100 * top1 / total, 100 * top5 / total
            results.append([run_name, epoch, acc1, acc5])
            print(f"  Epoch {epoch}: Top-1 {acc1:.2f}%, Top-5 {acc5:.2f}%")

    # -------------------------
    # Save results to CSV
    # -------------------------
    with open(args.csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run", "epoch", "top1", "top5"])
        writer.writerows(results)

    print(f"\nAll results saved to {args.csv_file}")
