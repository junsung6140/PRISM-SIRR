# PRISM: Latent Composition Consistency for Single-Image Reflection Removal

Official code repository for our **ECCV 2026** paper:

> **PRISM: Latent Composition Consistency for Single-Image Reflection Removal**
> Junseong Shin, Tae Hyun Kim
> VILAB, Hanyang University
> *European Conference on Computer Vision (ECCV), 2026*

🌐 **Project page:** https://junsung6140.github.io/prism/

## Abstract

Single-image reflection removal (SIRR) seeks to recover the transmission layer from a mixture corrupted by reflections — a severely ill-posed problem. Existing methods operate in pixel space, where the nonlinear sRGB formation model entangles the two layers and limits generalization. We observe that pretrained VAE latent spaces naturally project transmission and reflection into nearly orthogonal subspaces, enabling a fundamentally more separable working space. Building on this finding, we propose **PRISM** (*Pretrained-latent Reflection Image Separation Model*), which reinterprets SIRR as a latent linear separation problem. Under an approximate additive formulation in latent space, PRISM learns a flow matching velocity field on a pretrained FLUX backbone that recovers both transmission and reflection in a single forward pass. To enforce robust disentanglement, we introduce a **Latent Composition Consistency (LCC)** strategy that constructs synthetic mixtures by swapping reflection latents across samples and enforces consistent decomposition via a cycle loss. We further propose a **Layer Contrastive Separation (LCS)** loss that promotes semantic separation between layers through patch-level contrastive learning, without requiring explicit reflection targets.

## Method overview

PRISM operates entirely in the **packed 128-channel FLUX VAE latent space**. Given a mixture image `I`, the FLUX transformer predicts a velocity field `v` such that

```
z_T = z_I + v          (transmission latent)
z_R = -v               (reflection latent)
```

Training combines four objectives:

| Loss | Purpose |
|------|---------|
| **Latent** `L_latent`     | L2 on velocity `v_pred` vs. `v_target = z_T - z_I` |
| **Pixel** `L_pixel`       | L1 + LPIPS on decoded transmission |
| **LCC** `L_cycle`         | Swap `z_R` across the batch, recompose, predict velocity again, and enforce cycle consistency in latent or pixel space |
| **LCS** `L_nce`           | Patch-level InfoNCE between predicted/GT transmission and reflection tokens |

## Repository layout

```
PRISM-SIRR/
├── flux_klein_rr.py                  # FluxKleinReflectionRemoval model
├── train_flux_klein_rr_cycle.py      # Training (LCC + LCS)
├── eval_flux_klein_rr.py             # Benchmark evaluation (PSNR/SSIM/LPIPS)
├── inference_flux_klein_rr.py        # Inference on arbitrary images
├── configs/
│   ├── flux_klein_rr_cycle.yaml      # Training config
│   ├── eval_flux_klein_rr.yaml       # Evaluation config
│   └── inference_flux_klein_rr.yaml  # Inference config
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   └── inference.sh
├── dataloader/
│   ├── dataset.py                    # SIRSDataset, PhysicalDataset, SIRSTestDataset, SynthesisDataset
│   ├── fusion.py                     # FusionDataset
│   ├── synthesis.py                  # RDNet + Physical reflection synthesis
│   └── transforms.py
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/junsung6140/PRISM-SIRR.git
cd PRISM-SIRR

# Recommended: a fresh Python 3.10+ environment
pip install -r requirements.txt
```

Core dependencies: `torch`, `diffusers`, `transformers`, `peft`, `accelerate`, `lpips`, `opencv-python`, `scikit-image`, `wandb`.

### Backbone weights

Download the [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) weights once (e.g. via `huggingface-cli download black-forest-labs/FLUX.2-klein-4B`) and point `model.backbone` in every config to that directory:

```yaml
model:
  backbone: "/your/path/to/FLUX.2-klein-4B"
```

## Data preparation

PRISM expects each SIRS-style dataset directory to follow this layout:

```
<dataset>/
├── blended/              # input mixture images (I)
├── transmission_layer/   # ground-truth transmission (T)
└── reflection_layer/     # optional ground-truth reflection (R)
                          # if missing, training falls back to R = I - T
```

The evaluation script expects standard SIRS benchmarks under one parent directory:

```
<test_root>/
├── real20_420/             # Zhang et al. (CVPR'18)
├── Nature/
└── SIR2/
    ├── PostcardDataset/
    ├── SolidObjectDataset/
    └── WildSceneDataset/
```

For on-the-fly synthesis training (`type: synthesis`), point `dir` at a folder of clean images (e.g. PASCAL VOC2012 JPEGImages); the loader randomly pairs them and runs RDNet-style or physically-motivated reflection synthesis.

## Training

1. Edit `configs/flux_klein_rr_cycle.yaml` and replace the `<PATH_TO_*>` placeholders with your local paths.
2. Launch:

```bash
bash scripts/train.sh
# or:
accelerate launch train_flux_klein_rr_cycle.py --config configs/flux_klein_rr_cycle.yaml
```

Defaults: 50k steps, full fine-tuning (LoRA toggle in config), bf16 mixed precision, gradient accumulation 2, cosine LR with linear warmup. Validation runs every 1k steps on `data.val_dir` and the best checkpoint by PSNR is saved as `best_model.pt`.

## Evaluation

1. Edit `configs/eval_flux_klein_rr.yaml` (set `checkpoint`, `test_dir`, `save_dir`).
2. Run:

```bash
bash scripts/eval.sh
# or:
python eval_flux_klein_rr.py --config configs/eval_flux_klein_rr.yaml \
    --checkpoint <PATH_TO_CHECKPOINT>/best_model.pt \
    --save_dir   <PATH_TO_OUTPUT>
```

Reports PSNR / SSIM / LPIPS per dataset and a summary across `Real`, `Nature`, and `SIR2` groups; predictions are written to `<save_dir>/<ckpt_name>/<dataset>/`.

To dump predictions without metric computation, add `--inference_only`.

## Inference on arbitrary images

```bash
bash scripts/inference.sh
# or:
python inference_flux_klein_rr.py --config configs/inference_flux_klein_rr.yaml \
    --checkpoint <PATH_TO_CHECKPOINT>/best_model.pt \
    --input_dir  <PATH_TO_IMAGES> \
    --save_dir   <PATH_TO_OUTPUT>
```

`input_dir` can be either a flat folder of images or a folder containing a `blended/` subfolder. Outputs are saved to `<save_dir>/<ckpt_name>/<name>/transmission/` and `.../reflection/`.

## Pretrained checkpoint

🚧 Coming soon — we will release the trained `best_model.pt` and a HuggingFace snapshot of the FLUX backbone configuration.

## Qualitative results

A collection of qualitative outputs (input mixture / predicted transmission / predicted reflection across the SIRS benchmarks) is available here:

📦 **[PRISM_results.zip (2.2 GB)](https://drive.google.com/file/d/1sj03RSPLXeLr30SNA-S-h_UGX84g0y5j/view?usp=sharing)**

## Citation

```bibtex
@inproceedings{shin2026prism,
    title={PRISM: Latent Composition Consistency for Single-Image Reflection Removal},
    author={Junseong Shin and Tae Hyun Kim},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2026}
}
```

## Acknowledgements

This codebase builds on the [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) backbone (released by Black Forest Labs), the `diffusers` library, and the SIRS evaluation protocol from Zhang et al. (CVPR'18). The reflection synthesis pipeline draws on [DSRNet / RDNet](https://github.com/mingcv/DSRNet) and standard physically-motivated formulations.

## License

This repository is released under the [Apache License 2.0](LICENSE). Note that the FLUX backbone and any pretrained weights are governed by their own respective licenses; please consult the upstream sources before redistribution.
