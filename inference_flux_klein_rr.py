"""
FLUX.2 Klein 4B - Reflection Removal Inference Script

Runs inference only and saves predicted transmission/reflection images.

Usage:
    python inference_flux_klein_rr.py \
        --config configs/inference_flux_klein_rr.yaml \
        --checkpoint /path/to/best_model.pt \
        --save_dir ./inference_results \
        --input_dir /path/to/images_or_dataset_dir
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flux_klein_rr import FluxKleinReflectionRemoval, Args as ModelArgs


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class InferenceDataset(Dataset):
    """Inference-only dataset.

    Supports both:
    - flat image directory: data_dir/*.jpg|png
    - dataset directory with blended/: data_dir/blended/*.jpg|png
    """

    def __init__(self, data_dir, align_size=32, fixed_size=None, source_key="source_image"):
        self.data_dir = data_dir
        self.align_size = align_size
        self.fixed_size = fixed_size
        self.source_key = source_key
        blended_dir = os.path.join(data_dir, "blended")
        self.image_dir = blended_dir if os.path.isdir(blended_dir) else data_dir

        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")

        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"[InferenceDataset] {len(self.filenames)} images from {self.image_dir}")

    def _align(self, img):
        if self.fixed_size is not None:
            if img.size != (self.fixed_size, self.fixed_size):
                img = img.resize((self.fixed_size, self.fixed_size), Image.BICUBIC)
        else:
            w, h = img.size
            new_w = (w // self.align_size) * self.align_size
            new_h = (h // self.align_size) * self.align_size
            if new_w != w or new_h != h:
                img = img.resize((new_w, new_h), Image.BICUBIC)
        return img

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        blended = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
        blended = self._align(blended)
        blended_t = TF.to_tensor(blended) * 2.0 - 1.0
        return {self.source_key: blended_t, "filename": fn}


@torch.no_grad()
def run_inference(model, dataloader, dataset_name, save_dir):
    model.set_eval()
    trans_dir = os.path.join(save_dir, "transmission")
    refl_dir = os.path.join(save_dir, "reflection")
    os.makedirs(trans_dir, exist_ok=True)
    os.makedirs(refl_dir, exist_ok=True)

    count = 0
    for batch in tqdm(dataloader, desc=f"Infer {dataset_name}"):
        source = batch[model.args.source_key]
        filename = batch.get("filename", ["unknown"])[0]

        pred_t = model.sample_transmission(source)
        pred_r = model.sample_reflection(source)
        pred_t_01 = ((pred_t + 1.0) / 2.0).clamp(0, 1)
        pred_r_01 = ((pred_r + 1.0) / 2.0).clamp(0, 1)

        base_name = Path(filename).stem
        TF.to_pil_image(pred_t_01[0].float().cpu()).save(os.path.join(trans_dir, f"{base_name}.png"))
        TF.to_pil_image(pred_r_01[0].float().cpu()).save(os.path.join(refl_dir, f"{base_name}.png"))
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Inference FLUX Klein Reflection Removal")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (overrides config)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Single input directory for inference (flat image dir or dir with blended/)")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Datasets to skip (e.g., --skip nature real20)")
    args = parser.parse_args()

    config = load_config(args.config)
    infer_config = config.get("inference", {})
    model_config = config["model"]
    data_config = config["data"]

    checkpoint = args.checkpoint or infer_config.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Checkpoint path must be specified via --checkpoint or in config inference.checkpoint")

    save_dir = args.save_dir or infer_config.get("save_dir", "./inference_results")
    input_dir = args.input_dir or infer_config.get("input_dir", None)
    test_dir = infer_config.get("test_dir", "/data5/junsung/datasets/reflection_removal/sirs/test")
    align_size = infer_config.get("align_size", 32)
    fixed_size = infer_config.get("fixed_size", None)
    num_workers = infer_config.get("num_workers", 4)

    model_args = ModelArgs(
        pretrained_model_name_or_path=model_config["backbone"],
        use_lora=model_config.get("use_lora", True),
        lora_rank=model_config.get("lora_rank", 32),
        lora_alpha=model_config.get("lora_alpha", 64),
        lora_dropout=model_config.get("lora_dropout", 0.05),
        source_key=data_config.get("source_key", "source_image"),
        target_key=data_config.get("target_key", "target_image"),
        reflection_key=data_config.get("reflection_key", "reflection_image"),
    )

    print(f"Loading model from {checkpoint}...")
    model = FluxKleinReflectionRemoval(model_args, mode="test", checkpoint_path=checkpoint)
    model.set_eval()

    dataset_map = {
        "real20": os.path.join(test_dir, "real20_420"),
        "nature": os.path.join(test_dir, "Nature"),
        "postcard": os.path.join(test_dir, "SIR2/PostcardDataset"),
        "solid": os.path.join(test_dir, "SIR2/SolidObjectDataset"),
        "wild": os.path.join(test_dir, "SIR2/WildSceneDataset"),
    }

    ckpt_name = Path(checkpoint).stem
    output_base = os.path.join(save_dir, ckpt_name)
    os.makedirs(output_base, exist_ok=True)

    def _looks_like_image_dir(path):
        if path is None or not os.path.isdir(path):
            return False
        entries = os.listdir(path)
        if "blended" in entries and os.path.isdir(os.path.join(path, "blended")):
            return True
        return any(e.lower().endswith((".png", ".jpg", ".jpeg")) for e in entries)

    if input_dir is None and _looks_like_image_dir(test_dir):
        input_dir = test_dir

    summary = {}
    if input_dir is not None:
        if not os.path.exists(input_dir):
            raise ValueError(f"input_dir not found: {input_dir}")
        ds_name = Path(os.path.normpath(input_dir)).name or "input"
        dataset = InferenceDataset(
            data_dir=input_dir,
            align_size=align_size,
            fixed_size=fixed_size,
            source_key=data_config.get("source_key", "source_image"),
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        ds_save_dir = os.path.join(output_base, ds_name)
        num_saved = run_inference(model, dataloader, ds_name, ds_save_dir)
        summary[ds_name] = num_saved
    else:
        for ds_name in [n for n in ["real20", "nature", "postcard", "solid", "wild"] if n not in args.skip]:
            ds_dir = dataset_map[ds_name]
            if not os.path.exists(ds_dir):
                print(f"Directory not found: {ds_dir}, skipping")
                continue

            dataset = InferenceDataset(
                data_dir=ds_dir,
                align_size=align_size,
                fixed_size=fixed_size,
                source_key=data_config.get("source_key", "source_image"),
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

            ds_save_dir = os.path.join(output_base, ds_name)
            num_saved = run_inference(model, dataloader, ds_name, ds_save_dir)
            summary[ds_name] = num_saved

    print("\n" + "=" * 70)
    print(f"Inference complete: {ckpt_name}")
    print("=" * 70)
    for ds_name, n in summary.items():
        print(f"{ds_name:<20} {'saved':>8} {n:>10}")
    print("=" * 70)

    summary_path = os.path.join(output_base, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Output base: {output_base}\n\n")
        f.write("Saved files:\n")
        f.write("  transmission/*.png\n")
        f.write("  reflection/*.png\n\n")
        f.write(f"{'Dataset':<20} {'Saved':>10}\n")
        f.write("-" * 32 + "\n")
        for ds_name, n in summary.items():
            f.write(f"{ds_name:<20} {n:>10}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
