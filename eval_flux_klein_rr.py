"""
FLUX.2 Klein 4B - Reflection Removal Evaluation Script

Evaluates on standard SIRS benchmarks:
  - real20 (Real20_420)
  - Nature
  - SIR2: Postcard, SolidObject, WildScene

Usage:
    python eval_flux_klein_rr.py \
        --config configs/flux_klein_rr_full.yaml \
        --checkpoint /path/to/best_model.pt \
        --save_dir ./results
"""

import argparse
import os
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

from flux_klein_rr import FluxKleinReflectionRemoval, Args as ModelArgs
from dataloader.dataset import SIRSTestDataset

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_psnr(pred, target):
    """Compute PSNR between two [0, 1] tensors"""
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + 1e-8)).item()


def compute_ssim(pred, target):
    """Compute SSIM between two [0, 1] tensors (single image)"""
    pred_np = pred[0].float().cpu().permute(1, 2, 0).numpy()
    target_np = target[0].float().cpu().permute(1, 2, 0).numpy()
    return compare_ssim(pred_np, target_np, channel_axis=2, data_range=1.0)


class InferenceDataset(Dataset):
    """Inference-only dataset that only reads blended images."""

    def __init__(self, data_dir, align_size=32, fixed_size=None, source_key="source_image"):
        self.data_dir = data_dir
        self.align_size = align_size
        self.fixed_size = fixed_size
        self.source_key = source_key
        self.blended_dir = os.path.join(data_dir, "blended")

        if not os.path.isdir(self.blended_dir):
            raise ValueError(f"blended/ directory not found: {self.blended_dir}")

        self.filenames = sorted([
            f for f in os.listdir(self.blended_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"[InferenceDataset] {len(self.filenames)} images from {self.blended_dir}")

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
        blended = Image.open(os.path.join(self.blended_dir, fn)).convert("RGB")
        blended = self._align(blended)
        blended_t = TF.to_tensor(blended) * 2.0 - 1.0
        return {self.source_key: blended_t, "filename": fn}


@torch.no_grad()
def evaluate_dataset(model, dataloader, dataset_name, save_dir=None, lpips_fn=None, inference_only=False):
    """Evaluate on a single dataset"""
    model.set_eval()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        metrics_file = None
        if not inference_only:
            metrics_file = open(os.path.join(save_dir, 'metrics.txt'), 'w')
            metrics_file.write("name,PSNR,SSIM,LPIPS\n")

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_samples = 0

    for batch in tqdm(dataloader, desc=f"Eval {dataset_name}"):
        source = batch[model.args.source_key]
        filename = batch.get("filename", ["unknown"])[0]

        # Inference
        pred = model.sample_transmission(source)

        # Convert prediction to [0, 1]
        pred_01 = ((pred + 1.0) / 2.0).clamp(0, 1)

        num_samples += 1

        # Save images
        if save_dir is not None:
            base_name = Path(filename).stem
            pred_pil = TF.to_pil_image(pred_01[0].float().cpu())
            pred_pil.save(os.path.join(save_dir, f"{base_name}.png"))

        if inference_only:
            continue

        target = batch[model.args.target_key]
        target_01 = ((target.to(pred.device) + 1.0) / 2.0).clamp(0, 1)

        # PSNR/SSIM/LPIPS
        psnr = compute_psnr(pred_01, target_01)
        ssim_val = compute_ssim(pred_01, target_01)
        lpips_val = 0.0
        if lpips_fn is not None:
            lpips_val = lpips_fn(pred.float(), target.to(pred.device).float()).item()
            total_lpips += lpips_val

        total_psnr += psnr
        total_ssim += ssim_val
        if metrics_file is not None:
            metrics_file.write(f"{filename},{psnr:.2f},{ssim_val:.4f},{lpips_val:.4f}\n")

    if save_dir is not None and metrics_file is not None:
        metrics_file.close()

    avg_psnr = total_psnr / max(num_samples, 1)
    avg_ssim = total_ssim / max(num_samples, 1)
    avg_lpips = total_lpips / max(num_samples, 1)

    return {
        'PSNR': avg_psnr,
        'SSIM': avg_ssim,
        'LPIPS': avg_lpips,
        'num_samples': num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FLUX Klein Reflection Removal")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--no_save', action='store_true', help='Skip saving prediction images')
    parser.add_argument('--inference_only', action='store_true',
                        help='Run model inference only (skip PSNR/SSIM/LPIPS and summary)')
    parser.add_argument('--skip', nargs='+', default=[],
                        help='Datasets to skip (e.g., --skip nature real20)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    eval_config = config.get('eval', {})

    # Resolve paths: CLI args override config
    checkpoint = args.checkpoint or eval_config.get('checkpoint')
    if checkpoint is None:
        raise ValueError("Checkpoint path must be specified via --checkpoint or in config eval.checkpoint")
    save_dir = args.save_dir or eval_config.get('save_dir', './eval_results')
    test_dir = eval_config.get('test_dir', '/data5/junsung/datasets/reflection_removal/sirs/test')
    align_size = eval_config.get('align_size', 32)
    inference_only = args.inference_only or eval_config.get('inference_only', False)

    # Create model
    model_config = config['model']
    data_config = config['data']

    model_args = ModelArgs(
        pretrained_model_name_or_path=model_config['backbone'],
        use_lora=model_config.get('use_lora', True),
        lora_rank=model_config.get('lora_rank', 32),
        lora_alpha=model_config.get('lora_alpha', 64),
        lora_dropout=model_config.get('lora_dropout', 0.05),
        source_key=data_config.get('source_key', 'source_image'),
        target_key=data_config.get('target_key', 'target_image'),
        reflection_key=data_config.get('reflection_key', 'reflection_image'),
    )

    print(f"Loading model from {checkpoint}...")
    model = FluxKleinReflectionRemoval(model_args, mode='test', checkpoint_path=checkpoint)
    model.set_eval()

    # LPIPS
    lpips_fn = None
    if not inference_only and LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='vgg').to(next(model.transformer.parameters()).device)
        lpips_fn.eval()

    # Dataset name -> directory mapping
    dataset_map = {
        'real20': os.path.join(test_dir, 'real20_420'),
        'nature': os.path.join(test_dir, 'Nature'),
        'postcard': os.path.join(test_dir, 'SIR2/PostcardDataset'),
        'solid': os.path.join(test_dir, 'SIR2/SolidObjectDataset'),
        'wild': os.path.join(test_dir, 'SIR2/WildSceneDataset'),
    }

    # Evaluation groups
    eval_groups = {
        'Real': ['real20'],
        'Nature': ['nature'],
        'SIR2': ['postcard', 'solid', 'wild'],
    }

    # Create output directory
    ckpt_name = Path(checkpoint).stem
    output_base = os.path.join(save_dir, ckpt_name)
    os.makedirs(output_base, exist_ok=True)

    # Evaluate all datasets
    all_results = {}
    for ds_name in [n for n in ['real20', 'nature', 'postcard', 'solid', 'wild'] if n not in args.skip]:
        ds_dir = dataset_map[ds_name]
        if not os.path.exists(ds_dir):
            print(f"Directory not found: {ds_dir}, skipping")
            continue

        if inference_only:
            dataset = InferenceDataset(
                data_dir=ds_dir,
                align_size=align_size,
                source_key=data_config.get('source_key', 'source_image'),
            )
        else:
            dataset = SIRSTestDataset(
                data_dir=ds_dir,
                align_size=align_size,
                source_key=data_config.get('source_key', 'source_image'),
                target_key=data_config.get('target_key', 'target_image'),
                reflection_key=data_config.get('reflection_key', 'reflection_image'),
            )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        no_save = args.no_save or not eval_config.get('save_images', True)
        ds_save_dir = None if no_save else os.path.join(output_base, ds_name)
        results = evaluate_dataset(
            model,
            dataloader,
            ds_name,
            save_dir=ds_save_dir,
            lpips_fn=lpips_fn,
            inference_only=inference_only,
        )
        all_results[ds_name] = results

    if inference_only:
        print("\n" + "=" * 70)
        print(f"Inference-only complete: {ckpt_name}")
        print("=" * 70)
        for ds_name, res in all_results.items():
            print(f"{ds_name:<20} {'saved':>8} {'-':>8} {'-':>8} {res['num_samples']:>5}")
        print("=" * 70)
        return

    # Print summary by group
    print("\n" + "=" * 70)
    print(f"Evaluation Results: {ckpt_name}")
    print("=" * 70)
    print(f"{'Dataset':<20} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'N':>5}")
    print("-" * 70)

    for group_name, ds_names in eval_groups.items():
        group_results = {k: v for k, v in all_results.items() if k in ds_names}
        if not group_results:
            continue

        # Print individual datasets in the group
        for ds_name, res in group_results.items():
            label = f"  {ds_name}" if len(ds_names) > 1 else f"{group_name}"
            print(f"{label:<20} {res['PSNR']:>8.2f} {res['SSIM']:>8.4f} {res['LPIPS']:>8.4f} {res['num_samples']:>5}")

        # Print group average if multiple datasets
        if len(group_results) > 1:
            total_n = sum(r['num_samples'] for r in group_results.values())
            avg_psnr = sum(r['PSNR'] * r['num_samples'] for r in group_results.values()) / total_n
            avg_ssim = sum(r['SSIM'] * r['num_samples'] for r in group_results.values()) / total_n
            avg_lpips = sum(r['LPIPS'] * r['num_samples'] for r in group_results.values()) / total_n
            print(f"{group_name + ' (avg)':<20} {avg_psnr:>8.2f} {avg_ssim:>8.4f} {avg_lpips:>8.4f} {total_n:>5}")
            all_results[f'{group_name}_avg'] = {'PSNR': avg_psnr, 'SSIM': avg_ssim, 'LPIPS': avg_lpips, 'num_samples': total_n}

        print("-" * 70)

    print("=" * 70)

    # Save summary
    summary_path = os.path.join(output_base, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Checkpoint: {checkpoint}\n\n")
        f.write(f"{'Dataset':<20} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'N':>5}\n")
        f.write("-" * 70 + "\n")
        for ds_name, res in all_results.items():
            f.write(f"{ds_name:<20} {res['PSNR']:>8.2f} {res['SSIM']:>8.4f} {res['LPIPS']:>8.4f} {res['num_samples']:>5}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
