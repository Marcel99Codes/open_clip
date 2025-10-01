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

checkpoint_config_medium_small =[
    {
        "path": "/data1/marcel/clip/models_dc_medium/rgb_small/2025_09_14-10_54_13-model_ViT-B-16-gray-lr_5e-05-b_512-j_2-p_amp/checkpoints",
        "colorspace": "rgb", 
        "pipeline": "single",
        "conv1_patch_size": 16,
        "convb_patch_size": 16,
        "grayscale_only": False
    },
    {
        "path": "/data1/marcel/clip/models_dc_medium/ycbcr_small/2025_09_14-10_50_33-model_ViT-B-16-gray-lr_5e-05-b_512-j_4-p_amp/checkpoints",
        "colorspace": "ycbcr", 
        "pipeline": "dual_c1",
        "conv1_patch_size": 16,
        "convb_patch_size": 56,
        "grayscale_only": False
    },
]


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


def load_data_and_classes(dataset_name, preprocess, class_subset=None):
    if dataset_name == "imagenet1k":
        dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="validation")
        class_names = dataset.features["label"].names
        class_indices = list(range(len(class_names)))
        key = "label"

    elif dataset_name == "imagenet_a":
        dataset = load_dataset("clip-benchmark/wds_imagenet-a", split="test", streaming=True)
        key = "cls"
        class_indices = list(range(200))
        class_names = [
            "stingray", "goldfinch", "junco", "American robin", "jay", "bald eagle", "vulture",
            "newt", "American bullfrog", "box turtle", "green iguana", "agama", "chameleon",
            "American alligator", "garter snake", "harvestman", "scorpion", "tarantula", "centipede",
            "sulphur-crested cockatoo", "lorikeet", "hummingbird", "toucan", "duck", "goose", "koala",
            "jellyfish", "sea anemone", "flatworm", "snail", "crayfish", "hermit crab", "flamingo",
            "great egret", "oystercatcher", "pelican", "sea lion", "Chihuahua", "Golden Retriever",
            "Rottweiler", "German Shepherd Dog", "pug", "red fox", "Persian cat", "lynx", "lion",
            "American black bear", "mongoose", "ladybug", "rhinoceros beetle", "weevil", "fly", "bee",
            "ant", "grasshopper", "stick insect", "cockroach", "praying mantis", "leafhopper", "dragonfly",
            "monarch butterfly", "small white butterfly", "gossamer-winged butterfly", "starfish",
            "cottontail rabbit", "porcupine", "fox squirrel", "marmot", "bison", "skunk", "armadillo",
            "baboon", "white-headed capuchin", "African bush elephant", "pufferfish", "academic gown",
            "accordion", "acoustic guitar", "airliner", "ambulance", "apron", "balance beam", "balloon",
            "banjo", "barn", "wheelbarrow", "basketball", "lighthouse", "beaker", "bikini", "hunting bow",
            "bow tie", "breastplate", "broom", "candle", "canoe", "castle", "cello", "chain",
            "storage chest", "Christmas stocking", "cowboy boot", "cradle", "rotary dial telephone",
            "digital clock", "doormat", "drumstick", "dumbbell", "envelope", "feather boa", "flagpole",
            "forklift", "fountain", "garbage truck", "goblet", "go-kart", "golf cart", "grand piano",
            "hair dryer", "clothes iron", "carved pumpkin", "jeep", "kimono", "lighter", "limousine",
            "manhole cover", "maraca", "marimba", "mask", "mitten", "mosque", "metal nail", "obelisk",
            "ocarina", "pipe organ", "parachute", "parking meter", "piggy bank", "pool table",
            "hockey puck", "quill", "racket", "fishing casting reel", "revolver", "rocking chair",
            "rugby ball", "salt shaker", "sandal", "saxophone", "school bus", "schooner",
            "sewing machine", "shovel", "sleeping bag", "snowmobile", "snowplow", "soap dispenser",
            "spatula", "spider web", "steam locomotive", "stethoscope", "couch", "submarine", "sundial",
            "suspension bridge", "syringe", "tank", "teddy bear", "toaster", "torch", "tricycle",
            "umbrella", "unicycle", "viaduct", "volleyball", "washing machine", "water tower",
            "wine bottle", "shipwreck", "guacamole", "pretzel", "cheeseburger", "hot dog", "broccoli",
            "cucumber", "bell pepper", "mushroom", "lemon", "banana", "cherimoya (custard apple)",
            "pomegranate", "carbonara", "bubble", "cliff", "volcano", "baseball player", "rapeseed",
            "yellow lady's slipper", "corn", "acorn"
        ]

    elif dataset_name == "imagenet_o":
        dataset = load_dataset("clip-benchmark/wds_imagenet-o", split="test", streaming=True)
        key = "cls"
        class_indices = list(range(200))
        class_names = [
            "goldfish", "triceratops", "harvestman", "centipede", "sulphur-crested cockatoo", "lorikeet",
            "jellyfish", "brain coral", "chambered nautilus", "dugong", "starfish", "sea urchin", "pig",
            "armadillo", "rock beauty fish", "pufferfish", "abacus", "accordion", "apron", "balance beam",
            "ballpoint pen", "Band-Aid", "banjo", "barbershop", "bath towel", "military hat (bearskin or shako)",
            "binoculars", "bolo tie", "bottle cap", "bra", "broom", "buckle", "bulletproof vest", "candle",
            "car mirror", "chain-link fence", "chainsaw", "bell or wind chime", "Christmas stocking",
            "movie theater", "combination lock", "corkscrew", "construction crane", "croquet ball", "dam",
            "digital clock", "dishcloth", "dog sled", "doormat", "drilling rig", "electric fan", "envelope",
            "espresso machine", "face powder", "feather boa", "fireboat", "fire screen", "flute", "folding chair",
            "fountain", "fountain pen", "frying pan", "golf ball", "greenhouse", "guillotine", "hamper",
            "hair dryer", "harmonica", "honeycomb", "hourglass", "clothes iron", "carved pumpkin", "jigsaw puzzle",
            "joystick", "lawn mower", "library", "lighter", "lipstick", "loupe magnifying glass",
            "magnetic compass", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze",
            "medicine cabinet", "mortar and pestle", "mosquito net", "mousetrap", "metal nail", "neck brace",
            "necklace", "baby pacifier", "ocarina", "oil filter", "pipe organ", "oscilloscope", "oxygen mask",
            "paddle wheel", "pan flute", "park bench", "pencil sharpener", "Petri dish", "plectrum", "picket fence",
            "pill bottle", "ping-pong ball", "pinwheel", "plate rack", "plunger", "pool table", "plant pot",
            "power drill", "prayer rug", "prison", "punching bag", "quill", "radiator", "fishing casting reel",
            "remote control", "eraser", "ruler measuring stick", "safe", "safety pin", "salt shaker",
            "weighing scale", "screw", "screwdriver", "shoji screen / room divider", "shopping cart",
            "shower cap", "shower curtain", "ski", "sleeping bag", "slot machine", "snowmobile",
            "soap dispenser", "solar thermal collector", "space heater", "spatula", "spider web", "stove",
            "strainer", "stretcher", "submarine", "swim trunks / shorts", "swing", "electrical switch",
            "syringe", "tennis ball", "thatched roof", "front curtain", "thimble", "throne", "tile roof",
            "toaster", "tricycle", "turnstile", "umbrella", "vending machine", "waffle iron", "washing machine",
            "water bottle", "water tower", "whistle", "Windsor tie", "wooden spoon", "wool", "crossword",
            "traffic light", "popsicle", "bagel", "pretzel", "hot dog", "mashed potatoes", "broccoli",
            "cauliflower", "zucchini", "acorn squash", "cucumber", "bell pepper", "Granny Smith apple",
            "strawberry", "orange", "lemon", "pineapple", "banana", "jackfruit", "pomegranate",
            "chocolate syrup", "meatloaf", "pizza", "burrito", "bubble", "volcano", "corn", "acorn",
            "hen of the woods mushroom"
        ]

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if class_subset is not None:
        indices = [int(i) for i in class_subset.split(",")]
        class_indices = [i for i in indices]
        valid_set = set(indices)
        dataset = dataset.filter(lambda x: x[key] in valid_set)

        idx_map = {old: new for new, old in enumerate(indices)}
        dataset = dataset.map(lambda x: {key: idx_map[x[key]]})

    return dataset, class_names, class_indices


def evaluate(dataset, class_names, text_features, batch_size=64, preprocess=None, device="cpu"):
    top1, top5, total = 0, 0, 0
    images, labels = [], []

    for sample in tqdm(dataset, desc="Evaluating", leave=False):
        key = "image" if "image" in sample else "jpg"
        images.append(preprocess(sample[key]))

        if "cls" in sample:
            labels.append(int(sample["cls"]))
        else:
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

    args = parser.parse_args()

    results = []

    for run_idx, ckpt_config in enumerate(checkpoint_config_medium):
        run_name = f"run_{run_idx+1}"
        print(f"\nEvaluating model: {run_name}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "ViT-B-16"

        open_clip.transformer_args.set_transformer_args(
            ckpt_config["pipeline"], ckpt_config["conv1_patch_size"],
            ckpt_config["convb_patch_size"], ckpt_config["grayscale_only"]
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None, colorspace=ckpt_config["colorspace"]
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device)

        dataset, class_names, class_indices = load_data_and_classes(
            args.dataset, preprocess, args.class_subset
        )

        for epoch in range(1, args.epochs + 1):
            ckpt_path = os.path.join(ckpt_config["path"], f"epoch_{epoch}.pt")
            print(f"  Loading checkpoint: {ckpt_path}")
            print(ckpt_config)

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
                'a photo of my {}.',
                'the plastic {}.',
                'a photo of the cool {}.',
                'a close-up photo of a {}.',
                'a black and white photo of the {}.',
                'a painting of the {}.',
                'a painting of a {}.',
                'a pixelated photo of the {}.',
                'a sculpture of the {}.',
                'a bright photo of the {}.',
                'a cropped photo of a {}.',
                'a plastic {}.',
                'a photo of the dirty {}.',
                'a jpeg corrupted photo of a {}.',
                'a blurry photo of the {}.',
                'a photo of the {}.',
                'a good photo of the {}.',
                'a rendering of the {}.',
                'a {} in a video game.',
                'a photo of one {}.',
                'a doodle of a {}.',
                'a close-up photo of the {}.',
                'a photo of a {}.',
                'the origami {}.',
                'the {} in a video game.',
                'a sketch of a {}.',
                'a doodle of the {}.',
                'a origami {}.',
                'a low resolution photo of a {}.',
                'the toy {}.',
                'a rendition of the {}.',
                'a photo of the clean {}.',
                'a photo of a large {}.',
                'a rendition of a {}.',
                'a photo of a nice {}.',
                'a photo of a weird {}.',
                'a blurry photo of a {}.',
                'a cartoon {}.',
                'art of a {}.',
                'a sketch of the {}.',
                'a embroidered {}.',
                'a pixelated photo of a {}.',
                'itap of the {}.',
                'a jpeg corrupted photo of the {}.',
                'a good photo of a {}.',
                'a plushie {}.',
                'a photo of the nice {}.',
                'a photo of the small {}.',
                'a photo of the weird {}.',
                'the cartoon {}.',
                'art of the {}.',
                'a drawing of the {}.',
                'a photo of the large {}.',
                'a black and white photo of a {}.',
                'the plushie {}.',
                'a dark photo of a {}.',
                'itap of a {}.',
                'graffiti of the {}.',
                'a toy {}.',
                'itap of my {}.',
                'a photo of a cool {}.',
                'a photo of a small {}.',
                'a tattoo of the {}.',
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
