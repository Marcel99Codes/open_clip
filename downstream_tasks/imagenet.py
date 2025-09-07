import argparse
import torch
import open_clip
from datasets import load_dataset
from tqdm import tqdm
import os


# -------------------------
# Dataset Loader
# -------------------------
def load_data_and_classes(dataset_name, preprocess, tokenizer, device, class_subset=None):
    if dataset_name == "imagenet1k":
        dataset = load_dataset("imagenet-1k", split="validation")
        labels = dataset.features["label"].names
        class_names = [labels[i].replace("_", " ") for i in range(len(labels))]

    elif dataset_name == "imagenet_a":
        dataset = load_dataset("Voxel51/ImageNet-A", split="test")
        class_names = sorted(set(dataset["label_name"]))

    elif dataset_name == "imagenet_c":
        dataset = load_dataset("imagenet-c", split="validation")
        labels = dataset.features["label"].names
        class_names = [labels[i].replace("_", " ") for i in range(len(labels))]

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Filter to subset if requested
    if class_subset is not None:
        if dataset_name == "imagenet_a":
            # class_subset is list of names
            class_subset = [c.strip() for c in class_subset.split(",")]
            class_names = [c for c in class_names if c in class_subset]
            dataset = dataset.filter(lambda x: x["label_name"] in class_names)
        else:
            # class_subset is list of indices
            indices = [int(i) for i in class_subset.split(",")]
            class_names = [class_names[i] for i in indices]
            dataset = dataset.filter(lambda x: x["label"] in indices)

    # Precompute text features
    text_inputs = tokenizer([f"a photo of a {c}" for c in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return dataset, class_names, text_features



# -------------------------
# Evaluation
# -------------------------
def evaluate(dataset, class_names, text_features, batch_size=64):
    top1, top5, total = 0, 0, 0
    images, labels = [], []

    for sample in tqdm(dataset, desc="Evaluating", leave=False):
        images.append(preprocess(sample["image"]))

        if "label_name" in sample:
            # ImageNet-A has label_name
            labels.append(class_names.index(sample["label_name"]))
        else:
            # ImageNet-1k and ImageNet-C have integer labels
            labels.append(sample["label"])

        if len(images) == batch_size:
            imgs = torch.stack(images).to(device)
            labs = torch.tensor(labels)

            with torch.no_grad():
                image_features = model.encode_image(imgs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            values, indices = logits.topk(5, dim=-1)

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
                        choices=["imagenet1k", "imagenet_a", "imagenet_c"],
                        help="Dataset to evaluate on")
    parser.add_argument("--checkpoints", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs/checkpoints to evaluate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--class_subset", type=str, default=None,
                        help="Comma-separated list of classes to evaluate, e.g., '0,5,10' or 'goldfish,great white shark'")

    args = parser.parse_args()

    # Device + model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B-16"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=None
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)

    # Load dataset + text features
    dataset, class_names, text_features = load_data_and_classes(
        args.dataset, preprocess, tokenizer, device
    )

    # Evaluate each checkpoint
    results = {}
    for epoch in range(1, args.epochs + 1):
        ckpt_path = os.path.join(args.checkpoints, f"epoch_{epoch}.pt")
        print(f"\nLoading checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        top1, top5, total = evaluate(dataset, class_names, text_features, batch_size=args.batch_size)
        acc1, acc5 = 100 * top1 / total, 100 * top5 / total
        results[epoch] = (acc1, acc5)

        print(f"Epoch {epoch}: Top-1 {acc1:.2f}%, Top-5 {acc5:.2f}%")

    # Final summary
    print("\nSummary of results:")
    for epoch, (acc1, acc5) in results.items():
        print(f"Epoch {epoch:02d}: Top-1 {acc1:.2f}%, Top-5 {acc5:.2f}%")
