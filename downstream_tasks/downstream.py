import argparse
import torch
import sys, os
from datasets import load_dataset
from tqdm import tqdm
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import open_clip

os.environ["HF_HUB_ENABLE_XET"] = "0"

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
    {
        "path": "/data1/marcel/clip/models_dc_medium/ycbcr/2025_09_11-15_26_52-model_ViT-B-16-lr_5e-05-b_512-j_2-p_amp/checkpoints",
        "colorspace": "ycbcr", 
        "pipeline": "single",
        "conv1_patch_size": 16,
        "convb_patch_size": 16,
        "grayscale_only": False
    },
]


def clean_state_dict(state_dict):
    # Remove the module. prefix if present
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v
    return new_sd

def load_data_and_classes(dataset_name):
    if dataset_name == "imagenet1k":
        dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="validation")
        key = "label"
        class_names = dataset.features[key].names
        class_indices = list(range(len(class_names)))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset, class_names, class_indices


def evaluate(dataset, class_names, text_features, batch_size=64, preprocess=None, device="cpu"):
    top1, top5, total = 0, 0, 0
    images, labels = [], []

    # Tqdm wraps iterable with a progress bar
    for sample in tqdm(dataset, desc="Evaluating", leave=False):
        images.append(preprocess(sample["image"]))
        labels.append(int(sample["label"]))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["imagenet1k", "imagenet_a", "imagenet_c", "imagenet_o"],
                        help="Dataset to evaluate on")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs/checkpoints to evaluate")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--class_subset", type=str, default=None,
                        help="Comma-separated list of classes to evaluate")
    parser.add_argument("--csv_file", type=str, default="evaluation_results.csv",
                        help="CSV file to save results")
    parser.add_argument("--model_name", type=str, default="ViT-B-16",
                        help="Name of the model to evaluate")

    args = parser.parse_args()

    results = []
    for run_idx, ckpt_config in enumerate(checkpoint_config_medium):
        run_name = f"run_{run_idx+1}"
        print(f"\nEvaluating model: {run_name}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        open_clip.transformer_args.set_transformer_args(
            ckpt_config["pipeline"], ckpt_config["conv1_patch_size"],
            ckpt_config["convb_patch_size"], ckpt_config["grayscale_only"]
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, pretrained=None, colorspace=ckpt_config["colorspace"]
        )
        tokenizer = open_clip.get_tokenizer(args.model_name)
        model = model.to(device)

        dataset, class_names, class_indices = load_data_and_classes(args.dataset)

        for epoch in range(1, args.epochs + 1):
            ckpt_path = os.path.join(ckpt_config["path"], f"epoch_{epoch}.pt")
            print(f"  Loading checkpoint: {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("ema_state_dict", checkpoint.get("state_dict", checkpoint))
            state_dict = clean_state_dict(state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Epoch {epoch}: missing {len(missing)} keys, unexpected {len(unexpected)} keys")

            model = model.to(device)
            model.eval()

            templates = [
                'a bad photo of a {}.',
                'a photo of many {}.',
                'a sculpture of a {}.',
                'a photo of the hard to see {}.',
                'a low resolution photo of the {}.',
                'a rendering of a {}.',
                'graffiti of a {}.',
                'a bad photo of the {}.',
                'a cropped photo of the {}.',
                'a tattoo of a {}.',
                'the embroidered {}.',
                'a photo of a hard to see {}.',
                'a bright photo of a {}.',
                'a photo of a clean {}.',
                'a photo of a dirty {}.',
                'a dark photo of the {}.',
                'a drawing of a {}.',
            ]

            text_features_list = []
            for template in templates:
                texts = [template.format(c) for c in class_names]
                text_inputs = tokenizer(texts).to(device)
                with torch.no_grad():
                    class_embeddings = model.encode_text(text_inputs)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    text_features_list.append(class_embeddings)

            text_features = torch.stack(text_features_list).mean(dim=0)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Evaluate
            top1, top5, total = evaluate(
                dataset, class_names, text_features,
                batch_size=args.batch_size, preprocess=preprocess, device=device
            )
            acc1, acc5 = 100 * top1 / total, 100 * top5 / total
            results.append([run_name, epoch, acc1, acc5])
            print(f"  Epoch {epoch}: Top-1 {acc1:.2f}%, Top-5 {acc5:.2f}%")

    with open(args.csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run", "epoch", "top1", "top5"])
        writer.writerows(results)

    print(f"\nAll results saved to {args.csv_file}")
