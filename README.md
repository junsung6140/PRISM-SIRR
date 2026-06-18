# PRISM: Latent Composition Consistency for Single-Image Reflection Removal

Official code repository for our **ECCV 2026** paper:

> **PRISM: Latent Composition Consistency for Single-Image Reflection Removal**
> Junseong Shin, Tae Hyun Kim
> VILAB, Hanyang University
> *European Conference on Computer Vision (ECCV), 2026*

🌐 **Project page:** https://junsung6140.github.io/prism/

## Abstract

Single-image reflection removal (SIRR) seeks to recover the transmission layer from a mixture corrupted by reflections — a severely ill-posed problem. Existing methods operate in pixel space, where the nonlinear sRGB formation model entangles the two layers and limits generalization. We observe that pretrained VAE latent spaces naturally project transmission and reflection into nearly orthogonal subspaces, enabling a fundamentally more separable working space. Building on this finding, we propose **PRISM** (*Pretrained-latent Reflection Image Separation Model*), which reinterprets SIRR as a latent linear separation problem. Under an approximate additive formulation in latent space, PRISM learns a flow matching velocity field on a pretrained FLUX backbone that recovers both transmission and reflection in a single forward pass. To enforce robust disentanglement, we introduce a **Latent Composition Consistency (LCC)** strategy that constructs synthetic mixtures by swapping reflection latents across samples and enforces consistent decomposition via a cycle loss. We further propose a **Layer Contrastive Separation (LCS)** loss that promotes semantic separation between layers through patch-level contrastive learning, without requiring explicit reflection targets.

## Status

🚧 **Code coming soon.** Stay tuned!

## Citation

```bibtex
@inproceedings{shin2026prism,
    title={PRISM: Latent Composition Consistency for Single-Image Reflection Removal},
    author={Junseong Shin and Tae Hyun Kim},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2026}
}
```
