"""
PRISM Model: FLUX.2 Klein 4B based Reflection Removal

Single-step velocity prediction in pretrained FLUX VAE latent space:

    z_I = vae.encode(input_image)            # 32ch
    z_I_packed = pack(z_I)                   # 128ch
    v        = transformer(z_I_packed)
    z_T_pkd  = z_I_packed + v                # transmission latent (packed)
    z_R_pkd  = -v                            # reflection latent  (packed)
    T        = vae.decode(unpack(z_T_pkd))

Components
- FLUX VAE (frozen): 32-channel latents, 2x2 packing (32 -> 128).
- FLUX.2 Klein transformer with optional LoRA adapters.
- save/load helpers for LoRA-only or full-finetune checkpoints.

Usage
    # training
    model = FluxKleinReflectionRemoval(args, mode='train')
    model.set_train()
    output = model(batch)

    # inference
    model = FluxKleinReflectionRemoval(args, mode='test', checkpoint_path='ckpt.pt')
    transmission = model.sample_transmission(input_image)
"""

import torch
import torch.nn as nn
from diffusers.models import Flux2Transformer2DModel, AutoencoderKLFlux2
from peft import LoraConfig


def initialize_flux_transformer(args):
    """Initialize FLUX Klein transformer with optional LoRA adapters."""
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False)
    transformer.train()

    lora_target_modules = None

    if args.use_lora:
        lora_target_modules = [
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "proj_out",
        ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        transformer.add_adapter(lora_config, adapter_name="default")
        transformer.set_adapter(["default"])
    else:
        transformer.requires_grad_(True)

    transformer.enable_gradient_checkpointing()
    return transformer, lora_target_modules


class FluxKleinReflectionRemoval(nn.Module):
    """PRISM: FLUX.2 Klein based reflection removal model."""

    def __init__(self, args, mode='train', checkpoint_path=None):
        super().__init__()

        self.args = args
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading FLUX VAE from {args.pretrained_model_name_or_path}...")
        self.vae = AutoencoderKLFlux2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        if args.use_lora:
            print(f"Loading FLUX Transformer with LoRA (rank={args.lora_rank})...")
        else:
            print(f"Loading FLUX Transformer (full finetuning)...")
        self.transformer, self.lora_target_modules = initialize_flux_transformer(args)

        self.use_lora = args.use_lora
        self.lora_rank = getattr(args, 'lora_rank', None)
        self.lora_alpha = getattr(args, 'lora_alpha', None)

        self.latent_channels = self.vae.config.latent_channels  # 32 for FLUX 2
        self.downsampling_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        print(f"  VAE: {self.latent_channels}ch latents, {self.downsampling_factor}x downsampling")

        self.vae.to(self.device)
        self.transformer.to(self.device)

        if mode == 'test' and checkpoint_path is not None:
            self.load_ckpt(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")

    def set_train(self):
        self.transformer.train()
        if self.use_lora:
            for n, p in self.transformer.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
        else:
            self.transformer.requires_grad_(True)

    def set_eval(self):
        self.transformer.eval()

    @torch.no_grad()
    def encode(self, images, mode='sample'):
        """Encode images [-1, 1] to packed, BN-normalized latents (B, 128, H/16, W/16)."""
        images = images.to(dtype=torch.bfloat16, device=self.device)
        latent_dist = self.vae.encode(images).latent_dist

        if mode == 'sample':
            latents = latent_dist.sample()
        else:
            latents = latent_dist.mean

        # Patchify: 32ch -> 128ch (2x2 packing)
        latents = self._pack_latents(latents)

        # FLUX VAE applies BN to packed (128ch) latents
        assert latents.shape[1] == self.vae.bn.running_mean.numel(), (
            f"VAE BN dimension mismatch: latents have {latents.shape[1]} channels, "
            f"but BN has {self.vae.bn.running_mean.numel()} channels."
        )

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = (latents - bn_mean) / bn_std

        return latents

    def decode(self, latents):
        """Decode packed latents (B, 128, H/16, W/16) back to images [-1, 1]."""
        latents = latents.to(dtype=torch.bfloat16, device=self.device)

        assert latents.shape[1] == self.vae.bn.running_mean.numel(), (
            f"VAE BN dimension mismatch: latents have {latents.shape[1]} channels, "
            f"but BN has {self.vae.bn.running_mean.numel()} channels."
        )

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * bn_std + bn_mean

        latents = self._unpack_latents(latents)
        images = self.vae.decode(latents).sample
        return images.clamp(-1, 1)

    def _pack_latents(self, latents):
        """Pack 32-channel latents to 128-channel (2x2 spatial -> channel)."""
        B, C, H, W = latents.shape
        latents = latents.reshape(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
        return latents

    def _unpack_latents(self, latents):
        """Unpack 128-channel latents back to 32-channel."""
        B, C, H, W = latents.shape
        latents = latents.reshape(B, C // 4, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, H * 2, W * 2)
        return latents

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        """FLUX2 4D positional ids for image tokens: (T=0, H, W, L=0)."""
        t = torch.arange(1, device=device)
        h = torch.arange(height, device=device)
        w = torch.arange(width, device=device)
        l = torch.arange(1, device=device)
        latent_image_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
        return latent_image_ids.unsqueeze(0).expand(batch_size, -1, -1)

    def _prepare_text_ids(self, encoder_hidden_states, device, dtype):
        """FLUX2 4D positional ids for text tokens: (T=0, H=0, W=0, L)."""
        batch_size, seq_len, _ = encoder_hidden_states.shape
        t = torch.arange(1, device=device)
        h = torch.arange(1, device=device)
        w = torch.arange(1, device=device)
        l = torch.arange(seq_len, device=device)
        txt_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
        return txt_ids.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_velocity(self, z_mixed, timestep=None):
        """Predict velocity v on already-packed mixture latents."""
        batch_size = z_mixed.shape[0]
        H, W = z_mixed.shape[2], z_mixed.shape[3]

        # (B, C, H, W) -> (B, H*W, C)
        z_seq = z_mixed.permute(0, 2, 3, 1).reshape(batch_size, H * W, z_mixed.shape[1])

        # Empty conditioning (FLUX expects encoder_hidden_states)
        transformer = self.transformer.module if hasattr(self.transformer, 'module') else self.transformer
        joint_dim = getattr(transformer.config, 'joint_attention_dim', 7680)
        encoder_hidden_states = torch.zeros(
            batch_size, 1, joint_dim,
            device=z_seq.device, dtype=torch.bfloat16,
        )

        if timestep is None:
            timestep = torch.ones(batch_size, device=z_seq.device)
        elif isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=z_seq.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] != batch_size:
            timestep = timestep.expand(batch_size)

        img_ids = self._prepare_latent_image_ids(batch_size, H, W, z_seq.device, z_seq.dtype)
        txt_ids = self._prepare_text_ids(encoder_hidden_states, z_seq.device, z_seq.dtype)

        output = self.transformer(
            hidden_states=z_seq,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )
        result = output[0] if isinstance(output, tuple) else output

        C = result.shape[-1]
        velocity = result.reshape(batch_size, H, W, C).permute(0, 3, 1, 2)
        return velocity

    def forward(self, batch):
        """Training forward pass. Returns predicted/target velocity and component latents."""
        z_mixed = self.encode(batch['source_image'])
        z_trans = self.encode(batch['target_image'])
        z_refl = self.encode(batch[self.args.reflection_key])

        v_pred = self.predict_velocity(z_mixed)

        # v = z_T - z_I  ->  z_T = z_I + v,  z_R = -v
        v_target = z_trans - z_mixed

        z_pred_trans = z_mixed + v_pred
        z_pred_refl = -v_pred

        return {
            'v_pred': v_pred,
            'v_target': v_target,
            'z_mixed': z_mixed,
            'z_trans': z_trans,
            'z_refl': z_refl,
            'z_pred_trans': z_pred_trans,
            'z_pred_refl': z_pred_refl,
        }

    @torch.no_grad()
    def sample_transmission(self, images):
        """Sample clean transmission layer from mixed input."""
        z_mixed = self.encode(images)
        v_pred = self.predict_velocity(z_mixed)
        z_trans = z_mixed + v_pred
        return self.decode(z_trans)

    @torch.no_grad()
    def sample_reflection(self, images):
        """Sample reflection layer from mixed input. z_R = -v_pred."""
        z_mixed = self.encode(images)
        v_pred = self.predict_velocity(z_mixed)
        z_refl = -v_pred
        return self.decode(z_refl)

    def save_model(self, output_path):
        """Save checkpoint (LoRA-only weights if use_lora, otherwise full transformer)."""
        state_dict = {'use_lora': self.use_lora}

        if self.use_lora:
            state_dict.update({
                'lora_target_modules': self.lora_target_modules,
                'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha,
                'state_dict_transformer': {
                    k: v for k, v in self.transformer.state_dict().items()
                    if 'lora' in k
                },
            })
        else:
            state_dict['state_dict_transformer'] = self.transformer.state_dict()

        torch.save(state_dict, output_path)
        print(f"Saved checkpoint to {output_path}")

    def load_ckpt(self, checkpoint_path):
        """Load checkpoint; must match use_lora setting."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        use_lora_in_ckpt = ckpt.get('use_lora', True)

        if use_lora_in_ckpt and self.use_lora:
            for n, p in self.transformer.named_parameters():
                if 'lora' in n and n in ckpt['state_dict_transformer']:
                    p.data.copy_(ckpt['state_dict_transformer'][n])
        elif not use_lora_in_ckpt and not self.use_lora:
            self.transformer.load_state_dict(ckpt['state_dict_transformer'], strict=True)
        else:
            raise ValueError(
                f"Checkpoint mismatch: ckpt has use_lora={use_lora_in_ckpt}, "
                f"but model has use_lora={self.use_lora}"
            )

        print(f"Loaded {len(ckpt['state_dict_transformer'])} parameters")


class Args:
    """Arguments for FluxKleinReflectionRemoval."""

    def __init__(
        self,
        pretrained_model_name_or_path="<PATH_TO_FLUX_KLEIN_4B>",
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        source_key="source_image",
        target_key="target_image",
        reflection_key="reflection_image",
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.source_key = source_key
        self.target_key = target_key
        self.reflection_key = reflection_key
