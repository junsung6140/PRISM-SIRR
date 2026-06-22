"""
FLUX.2 Klein 4B - Reflection Removal Training Script
Swap-Compose-Cycle loss + InfoNCE for disentanglement

Usage:
    accelerate launch train_flux_klein_rr_cycle.py --config configs/flux_klein_rr_cycle.yaml
"""

import argparse
import yaml
import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as compare_ssim

# Import model
from flux_klein_rr import FluxKleinReflectionRemoval, Args as ModelArgs

# Import dataset
from dataloader.dataset import SIRSDataset, SIRSTestDataset, SynthesisDataset, PhysicalDataset
from dataloader.fusion import FusionDataset

# LPIPS for perceptual loss
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config):
    """Create training and validation datasets"""
    data_config = config['data']

    datasets = []
    ratios = []

    for ds_config in data_config['train_datasets']:
        ds_dir = ds_config['dir']
        ds_type = ds_config['type']
        ds_ratio = ds_config['ratio']

        if ds_ratio <= 0.0:
            logger.info(f"Skipping dataset: type={ds_type}, dir={ds_dir}, ratio={ds_ratio}")
            continue

        logger.info(f"Creating dataset: type={ds_type}, dir={ds_dir}, ratio={ds_ratio}")

        if ds_type == 'synthesis':
            dataset = SynthesisDataset(
                data_dir=ds_dir,
                patch_size=data_config['patch_size'],
                source_key=data_config['source_key'],
                target_key=data_config['target_key'],
                reflection_key=data_config['reflection_key'],
                synthesis_method=ds_config.get('synthesis_method', 'rdnet'),
            )
        elif ds_type == 'sirs':
            dataset = SIRSDataset(
                data_dir=ds_dir,
                patch_size=data_config['patch_size'],
                source_key=data_config['source_key'],
                target_key=data_config['target_key'],
                reflection_key=data_config['reflection_key'],
                load_reflection_layer=True,
            )
        elif ds_type == 'physical':
            dataset = PhysicalDataset(
                data_dir=ds_dir,
                patch_size=data_config['patch_size'],
                source_key=data_config['source_key'],
                target_key=data_config['target_key'],
                reflection_key=data_config['reflection_key'],
            )
        else:
            raise ValueError(f"Unknown dataset type: {ds_type}")

        datasets.append(dataset)
        ratios.append(ds_ratio)
        logger.info(f"Dataset created: {len(dataset)} samples")

    if data_config.get('use_fusion', True) and len(datasets) > 1:
        train_dataset = FusionDataset(
            datasets=datasets,
            fusion_ratios=ratios,
            size=data_config.get('fusion_size', 10000),
        )
        logger.info(f"Fusion dataset created: {len(train_dataset)} samples")
    else:
        train_dataset = datasets[0]

    val_dataset = SIRSTestDataset(
        data_dir=data_config['val_dir'],
        align_size=32,
        source_key=data_config['source_key'],
        target_key=data_config['target_key'],
        reflection_key=data_config['reflection_key'],
    )
    logger.info(f"Validation dataset created: {len(val_dataset)} samples")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config):
    """Create training and validation dataloaders"""
    data_config = config['data']

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True,
    )

    return train_loader, val_loader


class LPIPSLoss:
    """LPIPS perceptual loss"""
    def __init__(self, device, dtype=torch.float32):
        if LPIPS_AVAILABLE:
            self.lpips = lpips.LPIPS(net='vgg').to(device)
            self.lpips = self.lpips.to(dtype)
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
            self.dtype = dtype
        else:
            self.lpips = None
            self.dtype = dtype

    def __call__(self, pred, target):
        if self.lpips is None:
            return torch.tensor(0.0, device=pred.device)

        original_dtype = pred.dtype
        pred_converted = pred.to(self.dtype)
        target_converted = target.to(self.dtype)

        loss = self.lpips(pred_converted, target_converted).mean()
        return loss.to(original_dtype)


def compute_loss(model_output, batch, config, model, lpips_loss=None):
    """
    Compute training loss: latent + pixel + cycle + InfoNCE.

    Args:
        model_output: Output from model.forward()
        batch: Input batch
        config: Full config dict
        model: FluxKleinReflectionRemoval model
        lpips_loss: LPIPS loss function (optional)

    Returns:
        total_loss, loss_dict
    """
    loss_config = config['loss']

    # ---- 1. Latent space loss (same as baseline) ----
    v_pred = model_output['v_pred']
    v_target = model_output['v_target']

    v_pred_f32 = v_pred.float()
    v_target_f32 = v_target.float()

    if loss_config['latent_type'] == 'l1':
        latent_loss = F.l1_loss(v_pred_f32, v_target_f32)
    else:
        latent_loss = F.mse_loss(v_pred_f32, v_target_f32)

    # ---- 2. Pixel space loss (same as baseline) ----
    pixel_loss = torch.tensor(0.0, device=v_pred.device, dtype=torch.float32)
    lpips_val = torch.tensor(0.0, device=v_pred.device, dtype=torch.float32)
    l1_val = torch.tensor(0.0, device=v_pred.device, dtype=torch.float32)

    if loss_config['pixel_weight'] > 0:
        pred_images = decode_latents(model_output['z_pred_trans'], model)
        target_images = batch[model.args.target_key].to(pred_images.device).float()

        max_size = loss_config.get('pixel_max_size', 512)
        if pred_images.shape[-1] > max_size or pred_images.shape[-2] > max_size:
            pred_images = F.interpolate(pred_images, size=(max_size, max_size), mode='bilinear', align_corners=False)
            target_images = F.interpolate(target_images, size=(max_size, max_size), mode='bilinear', align_corners=False)

        if pred_images.dtype == torch.bfloat16:
            pred_images = pred_images.float()
            target_images = target_images.float()

        if 'lpips' in loss_config['pixel_type'] and lpips_loss is not None:
            lpips_val = lpips_loss(pred_images, target_images)
            pixel_loss = pixel_loss + lpips_val
        if 'l1' in loss_config['pixel_type']:
            l1_val = F.l1_loss(pred_images, target_images)
            pixel_loss = pixel_loss + l1_val

    # ---- 3. Swap-Compose-Cycle loss ----
    cycle_loss = torch.tensor(0.0, device=v_pred.device, dtype=torch.float32)
    cycle_weight = loss_config.get('cycle_weight', 0.0)

    nce_loss = torch.tensor(0.0, device=v_pred.device, dtype=torch.float32)
    nce_weight = loss_config.get('nce_weight', 0.0)

    if cycle_weight > 0 or nce_weight > 0:
        # Detached components from 1st forward
        z_T = model_output['z_pred_trans'].detach()  # sg(z_T)
        z_R = model_output['z_pred_refl'].detach()    # sg(z_R) = sg(-v)

        # Swap reflections within batch (roll by 1)
        z_R_swapped = z_R.roll(1, dims=0)

        # Compose swapped mixture
        z_I_swap = z_T + z_R_swapped

        # 2nd forward: predict velocity on swapped mixture
        v_cycle = model.predict_velocity(z_I_swap)

        # Cycle predictions
        z_T_cycle = z_I_swap + v_cycle       # should recover z_T
        z_R_cycle = -v_cycle                  # should recover z_R_swapped

        # Cycle consistency losses
        if cycle_weight > 0:
            cycle_mode = loss_config.get('cycle_mode', 'both')
            cycle_space = loss_config.get('cycle_space', 'latent')

            if cycle_space == 'pixel':
                cycle_max_size = loss_config.get('cycle_pixel_max_size', loss_config.get('pixel_max_size', 512))
                cycle_pixel_type = loss_config.get('cycle_pixel_type', 'l1')

                def _decode_for_cycle(z, with_grad):
                    if with_grad:
                        img = decode_latents(z, model)
                    else:
                        with torch.no_grad():
                            img = decode_latents(z, model)
                    if img.shape[-1] > cycle_max_size or img.shape[-2] > cycle_max_size:
                        img = F.interpolate(img, size=(cycle_max_size, cycle_max_size),
                                            mode='bilinear', align_corners=False)
                    if img.dtype == torch.bfloat16:
                        img = img.float()
                    return img

                def _pixel_cycle_loss(pred_img, tgt_img):
                    l = torch.tensor(0.0, device=pred_img.device, dtype=torch.float32)
                    if 'l1' in cycle_pixel_type:
                        l = l + F.l1_loss(pred_img, tgt_img)
                    if 'lpips' in cycle_pixel_type and lpips_loss is not None:
                        l = l + lpips_loss(pred_img, tgt_img)
                    return l

                if cycle_mode == 'T':
                    img_T_cycle = _decode_for_cycle(z_T_cycle, with_grad=True)
                    img_T = _decode_for_cycle(z_T, with_grad=False)
                    cycle_loss = _pixel_cycle_loss(img_T_cycle, img_T)
                elif cycle_mode == 'R':
                    img_R_cycle = _decode_for_cycle(z_R_cycle, with_grad=True)
                    img_R = _decode_for_cycle(z_R_swapped, with_grad=False)
                    cycle_loss = _pixel_cycle_loss(img_R_cycle, img_R)
                else:  # "both"
                    img_T_cycle = _decode_for_cycle(z_T_cycle, with_grad=True)
                    img_T = _decode_for_cycle(z_T, with_grad=False)
                    img_R_cycle = _decode_for_cycle(z_R_cycle, with_grad=True)
                    img_R = _decode_for_cycle(z_R_swapped, with_grad=False)
                    cycle_loss_T = _pixel_cycle_loss(img_T_cycle, img_T)
                    cycle_loss_R = _pixel_cycle_loss(img_R_cycle, img_R)
                    cycle_loss = (cycle_loss_T + cycle_loss_R) / 2.0
            else:
                if cycle_mode == 'T':
                    cycle_loss = F.l1_loss(z_T_cycle.float(), z_T.float())
                elif cycle_mode == 'R':
                    cycle_loss = F.l1_loss(z_R_cycle.float(), z_R_swapped.float())
                else:  # "both"
                    cycle_loss_T = F.l1_loss(z_T_cycle.float(), z_T.float())
                    cycle_loss_R = F.l1_loss(z_R_cycle.float(), z_R_swapped.float())
                    cycle_loss = (cycle_loss_T + cycle_loss_R) / 2.0

        # ---- 7. Swap-InfoNCE loss ----
        if nce_weight > 0:
            tau = loss_config.get('nce_tau', 0.1)
            nce_patch_size = loss_config.get('nce_patch_size', 0)  # 0 = global (GAP)

            # GT latents (detached, from VAE encode)
            z_gt_T = model_output['z_trans'].float().detach()
            z_gt_R = model_output['z_refl'].float().detach()

            def _to_patch_tokens(z, patch_sz):
                """(B, C, H, W) → (B*P, C) where P = num_patches per sample."""
                if patch_sz <= 0:
                    return z.mean(dim=[2, 3])  # global: (B, C)
                B, C, H, W = z.shape
                ph = H // patch_sz
                pw = W // patch_sz
                # trim to exact multiples
                z = z[:, :, :ph * patch_sz, :pw * patch_sz]
                # (B, C, ph, patch_sz, pw, patch_sz) → (B, ph, pw, C, patch_sz, patch_sz)
                z = z.reshape(B, C, ph, patch_sz, pw, patch_sz)
                z = z.permute(0, 2, 4, 1, 3, 5)  # (B, ph, pw, C, patch_sz, patch_sz)
                z = z.reshape(B * ph * pw, C, patch_sz, patch_sz)
                return z.mean(dim=[2, 3])  # (B*P, C)

            # Anchor: z_T_cycle
            anchor = F.normalize(_to_patch_tokens(z_T_cycle.float(), nce_patch_size), dim=1)  # (B*P, C)

            # Positives: sg(z_pred_T) + GT z_T
            pos_pred = F.normalize(_to_patch_tokens(z_T.float(), nce_patch_size), dim=1)       # (B*P, C)
            pos_gt = F.normalize(_to_patch_tokens(z_gt_T, nce_patch_size), dim=1)              # (B*P, C)

            # Negatives: z_pred_R + z_R_cycle(detached) + GT z_R (only if GT exists)
            neg_R = F.normalize(_to_patch_tokens(z_R.float(), nce_patch_size), dim=1)               # (B*P, C)
            neg_R_cycle = F.normalize(_to_patch_tokens(z_R_cycle.float().detach(), nce_patch_size), dim=1)  # (B*P, C)
            neg_list = [neg_R, neg_R_cycle]

            # Only use GT reflection as negative when it's real GT (not I-T fallback)
            has_gt = batch.get('has_gt_reflection', None)
            if has_gt is not None and has_gt.sum() > 0:
                gt_mask = has_gt.bool()
                neg_gt_R_all = F.normalize(_to_patch_tokens(z_gt_R, nce_patch_size), dim=1)  # (B*P, C)
                if nce_patch_size > 0:
                    B = z_gt_R.shape[0]
                    P = neg_gt_R_all.shape[0] // B
                    # expand mask to patch level: (B,) → (B*P,)
                    patch_mask = gt_mask.unsqueeze(1).expand(B, P).reshape(-1)
                    neg_list.append(neg_gt_R_all[patch_mask])
                else:
                    neg_list.append(neg_gt_R_all[gt_mask])

            all_negatives = torch.cat(neg_list, dim=0)

            # Positive similarities: anchor_i vs matched positive at same patch position
            pos_sim_pred = (anchor * pos_pred).sum(dim=1) / tau
            pos_sim_gt = (anchor * pos_gt).sum(dim=1) / tau       # (N,)
            pos_logits = torch.stack([pos_sim_pred, pos_sim_gt], dim=1)  # (N, 2)

            # Negative similarities: (N, N_neg)
            neg_sim = torch.mm(anchor, all_negatives.t()) / tau

            # InfoNCE with multiple positives:
            # -log( (exp(pos1)+exp(pos2)) / (exp(pos1)+exp(pos2)+Σexp(neg)) )
            all_logits = torch.cat([pos_logits, neg_sim], dim=1)  # (N, 2+N_neg)
            nce_loss = (-torch.logsumexp(pos_logits, dim=1) + torch.logsumexp(all_logits, dim=1)).mean()

    # ---- Total loss ----
    total_loss = (
        loss_config['latent_weight'] * latent_loss +
        loss_config['pixel_weight'] * pixel_loss +
        cycle_weight * cycle_loss +
        nce_weight * nce_loss
    )

    loss_dict = {
        'loss': total_loss.item(),
        'latent_loss': latent_loss.item(),
        'pixel_loss': pixel_loss.item(),
        'lpips_loss': lpips_val.item(),
        'l1_loss': l1_val.item(),
        'cycle_loss': cycle_loss.item(),
        'nce_loss': nce_loss.item(),
    }

    return total_loss, loss_dict


def decode_latents(latents, model):
    """
    Decode latents to images (gradient-enabled for pixel loss)
    """
    original_dtype = latents.dtype

    latents = latents.to(dtype=torch.bfloat16)

    bn_mean = model.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(
        model.vae.bn.running_var.view(1, -1, 1, 1) + model.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * bn_std + bn_mean

    latents = model._unpack_latents(latents)

    images = model.vae.decode(latents).sample

    images = images.to(original_dtype)
    return images.clamp(-1, 1)


@torch.no_grad()
def validate(model, val_loader, accelerator, lpips_loss=None):
    """Run validation"""
    model.set_eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_samples = 0

    for batch in tqdm(val_loader, desc="Validation", disable=not accelerator.is_local_main_process):
        source = batch[model.args.source_key]
        target = batch[model.args.target_key]

        pred = model.sample_transmission(source)

        pred_01 = ((pred + 1.0) / 2.0).clamp(0, 1)
        target_01 = ((target.to(pred.device) + 1.0) / 2.0).clamp(0, 1)

        mse = F.mse_loss(pred_01, target_01)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        total_psnr += psnr.item()

        pred_np = pred_01[0].float().cpu().permute(1, 2, 0).numpy()
        target_np = target_01[0].float().cpu().permute(1, 2, 0).numpy()
        ssim_val = compare_ssim(pred_np, target_np, channel_axis=2, data_range=1.0)
        total_ssim += ssim_val

        if lpips_loss is not None:
            lpips_val = lpips_loss(pred, target.to(pred.device))
            total_lpips += lpips_val.item()

        num_samples += 1

    avg_psnr = total_psnr / max(num_samples, 1)
    avg_ssim = total_ssim / max(num_samples, 1)
    avg_lpips = total_lpips / max(num_samples, 1)

    model.set_train()

    return {'val/psnr': avg_psnr, 'val/ssim': avg_ssim, 'val/lpips': avg_lpips}


@torch.no_grad()
def log_images(model, val_loader, accelerator, step, output_dir, num_images=20):
    """Save individual prediction images (no grid)"""
    model.set_eval()

    if not accelerator.is_main_process:
        model.set_train()
        return

    samples_dir = Path(output_dir) / "samples" / f"step_{step:06d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for batch_idx, batch in enumerate(val_loader):
        if count >= num_images:
            break

        source = batch[model.args.source_key]
        target = batch[model.args.target_key]
        filename = batch.get("filename", [f"img_{batch_idx}"])[0]

        pred_trans = model.sample_transmission(source)
        pred_refl = model.sample_reflection(source)

        source_vis = ((source[0] + 1.0) / 2.0).clamp(0, 1)
        target_vis = ((target[0] + 1.0) / 2.0).clamp(0, 1)
        pred_trans_vis = ((pred_trans[0] + 1.0) / 2.0).clamp(0, 1)
        pred_refl_vis = ((pred_refl[0] + 1.0) / 2.0).clamp(0, 1)

        base_name = Path(filename).stem
        TF.to_pil_image(source_vis.float().cpu()).save(samples_dir / f"{base_name}_input.png")
        TF.to_pil_image(target_vis.float().cpu()).save(samples_dir / f"{base_name}_target.png")
        TF.to_pil_image(pred_trans_vis.float().cpu()).save(samples_dir / f"{base_name}_pred_trans.png")
        TF.to_pil_image(pred_refl_vis.float().cpu()).save(samples_dir / f"{base_name}_pred_refl.png")

        count += 1

    logger.info(f"Saved {count} validation images to {samples_dir}")
    model.set_train()


def train(config, args):
    """Main training function"""

    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    mixed_precision = config['training'].get('mixed_precision', 'bf16')

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb" if args.use_wandb else None,
    )

    if args.seed is not None:
        set_seed(args.seed)

    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process and args.use_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            name=output_dir.name,
            config=config,
            dir=str(output_dir),
        )

    # Create model
    logger.info("Creating model...")
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

    model = FluxKleinReflectionRemoval(model_args, mode='train')
    model.set_train()

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    # Setup optimizer
    train_config = config['training']
    optimizer_config = train_config.get('optimizer_kwargs', {})

    trainable_params = []
    if model.use_lora:
        for n, p in model.transformer.named_parameters():
            if 'lora' in n and p.requires_grad:
                trainable_params.append(p)
    else:
        trainable_params = list(model.transformer.parameters())

    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config['learning_rate'],
        betas=optimizer_config.get('betas', [0.9, 0.999]),
        weight_decay=optimizer_config.get('weight_decay', 0.01),
        eps=optimizer_config.get('eps', 1e-8),
    )

    # Setup scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
    from torch.optim.lr_scheduler import SequentialLR

    warmup_steps = train_config.get('warmup_steps', 500)
    max_steps = train_config['max_steps']

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=train_config['learning_rate'] * 0.1,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    # Prepare with accelerator
    model.transformer, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model.transformer, optimizer, train_loader, val_loader, scheduler
    )

    model.vae = model.vae.to(accelerator.device)

    # Create LPIPS loss
    _need_lpips = (
        'lpips' in config['loss'].get('pixel_type', '') or
        'lpips' in config['loss'].get('cycle_pixel_type', '')
    )
    lpips_loss = LPIPSLoss(accelerator.device) if _need_lpips else None

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_psnr = 0.0

    progress_bar = tqdm(
        total=max_steps,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    while True:
        for batch in train_loader:
            with accelerator.accumulate(model.transformer):
                # Forward pass (1st forward)
                model_output = model(batch)

                # Compute loss (includes 2nd forward for cycle if enabled)
                loss, loss_dict = compute_loss(model_output, batch, config, model, lpips_loss)

                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % config['logging'].get('log_interval', 100) == 0:
                    if accelerator.is_main_process:
                        lr = optimizer.param_groups[0]['lr']
                        log_dict = {
                            'train/lr': lr,
                            'train/step': global_step,
                            **{f'train/{k}': v for k, v in loss_dict.items()}
                        }

                        if args.use_wandb and wandb.run is not None:
                            wandb.log(log_dict, step=global_step)

                        progress_bar.set_postfix(**{k.replace('train/', ''): f"{v:.4f}" for k, v in log_dict.items()})

                # Validation
                if global_step % train_config.get('validation_steps', 1000) == 0:
                    val_metrics = validate(model, val_loader, accelerator, lpips_loss=lpips_loss)

                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step}: "
                            f"PSNR = {val_metrics['val/psnr']:.2f} dB, "
                            f"SSIM = {val_metrics['val/ssim']:.4f}, "
                            f"LPIPS = {val_metrics['val/lpips']:.4f}"
                        )
                        if args.use_wandb and wandb.run is not None:
                            wandb.log(val_metrics, step=global_step)

                        if val_metrics['val/psnr'] > best_psnr:
                            best_psnr = val_metrics['val/psnr']
                            best_path = output_dir / "best_model.pt"
                            wrapped_transformer = model.transformer
                            model.transformer = accelerator.unwrap_model(model.transformer)
                            model.save_model(best_path)
                            model.transformer = wrapped_transformer
                            logger.info(f"New best PSNR: {best_psnr:.2f} dB -> saved to {best_path}")

                    if config['logging'].get('save_images', True):
                        log_images(model, val_loader, accelerator, global_step, output_dir,
                                 num_images=config['logging'].get('num_val_images', 4))

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    # Save final checkpoint
    if accelerator.is_main_process:
        final_path = output_dir / "final_model.pt"
        wrapped_transformer = model.transformer
        model.transformer = accelerator.unwrap_model(model.transformer)
        model.save_model(final_path)
        model.transformer = wrapped_transformer
        logger.info(f"Saved final model to {final_path}")

    if args.use_wandb and wandb.run is not None:
        wandb.finish()

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help='Disable wandb logging')
    parser.set_defaults(use_wandb=True)

    args = parser.parse_args()

    config = load_config(args.config)

    train(config, args)


if __name__ == "__main__":
    main()
