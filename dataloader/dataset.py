"""SIRS Dataset classes for training and testing."""

import os
import random
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .transforms import paired_transforms, paired_data_transforms


class SIRSDataset(Dataset):
    """SIRS training dataset.

    Loads paired (blended, transmission, reflection) data from
        data_dir/blended/
        data_dir/transmission_layer/
        data_dir/reflection_layer/   (optional; falls back to R = I - T)
    """

    def __init__(self, data_dir, patch_size=512, source_key="source_image",
                 target_key="target_image", reflection_key="reflection_image",
                 load_reflection_layer=False):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.source_key = source_key
        self.target_key = target_key
        self.reflection_key = reflection_key

        self.blended_dir = os.path.join(data_dir, "blended")
        self.trans_dir = os.path.join(data_dir, "transmission_layer")
        self.refl_dir = os.path.join(data_dir, "reflection_layer")

        refl_files = set()
        if os.path.exists(self.refl_dir):
            refl_files = set(os.listdir(self.refl_dir))
        self.has_gt_reflection = len(refl_files) > 0

        blended_files = set(os.listdir(self.blended_dir))
        trans_files = set(os.listdir(self.trans_dir)) if os.path.exists(self.trans_dir) else blended_files
        common = blended_files & trans_files
        if self.has_gt_reflection:
            common = common & refl_files

        exts = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        self.filenames = sorted([f for f in common if os.path.splitext(f)[1] in exts])

        refl_src = "reflection_layer/" if self.has_gt_reflection else "I - T (fallback)"
        print(f"[SIRSDataset] {len(self.filenames)} images from {data_dir}  (R source: {refl_src})")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]

        blended = Image.open(os.path.join(self.blended_dir, fn)).convert('RGB')
        trans = Image.open(os.path.join(self.trans_dir, fn)).convert('RGB')

        to_tensor = lambda img: TF.to_tensor(img) * 2.0 - 1.0

        if self.has_gt_reflection:
            refl = Image.open(os.path.join(self.refl_dir, fn)).convert('RGB')
            images = paired_transforms([blended, trans, refl], self.patch_size)
            blended_t = to_tensor(images[0])
            trans_t = to_tensor(images[1])
            refl_t = to_tensor(images[2])
        else:
            images = paired_transforms([blended, trans], self.patch_size)
            blended_t = to_tensor(images[0])
            trans_t = to_tensor(images[1])
            refl_t = blended_t - trans_t  # R = I - T fallback

        return {
            self.source_key: blended_t,
            self.target_key: trans_t,
            self.reflection_key: refl_t,
            "has_gt_reflection": torch.tensor(0.0),
            "filename": fn,
        }


class PhysicalDataset(Dataset):
    """Physical capture dataset (requires GT reflection layer).

    For physically-captured glass-reflection data the reflection cannot be
    recovered as I - T due to non-linear blending, so reflection_layer/ must
    exist for every sample.
    """

    def __init__(self, data_dir, patch_size=512, source_key="source_image",
                 target_key="target_image", reflection_key="reflection_image"):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.source_key = source_key
        self.target_key = target_key
        self.reflection_key = reflection_key

        self.blended_dir = os.path.join(data_dir, "blended")
        self.trans_dir = os.path.join(data_dir, "transmission_layer")
        self.refl_dir = os.path.join(data_dir, "reflection_layer")

        if not os.path.exists(self.refl_dir):
            raise ValueError(
                f"[PhysicalDataset] reflection_layer/ not found in {data_dir}. "
                f"Physical datasets require pre-captured GT reflections."
            )

        blended_files = set(os.listdir(self.blended_dir))
        trans_files = set(os.listdir(self.trans_dir))
        refl_files = set(os.listdir(self.refl_dir))

        common = blended_files & trans_files & refl_files
        exts = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        self.filenames = sorted([f for f in common if os.path.splitext(f)[1] in exts])

        if len(self.filenames) == 0:
            raise ValueError(
                f"[PhysicalDataset] No common files across blended/, "
                f"transmission_layer/, reflection_layer/ in {data_dir}"
            )

        print(f"[PhysicalDataset] {len(self.filenames)} images from {data_dir} (GT reflection)")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]

        blended = Image.open(os.path.join(self.blended_dir, fn)).convert('RGB')
        trans = Image.open(os.path.join(self.trans_dir, fn)).convert('RGB')
        refl = Image.open(os.path.join(self.refl_dir, fn)).convert('RGB')

        images = paired_transforms([blended, trans, refl], self.patch_size)
        to_tensor = lambda img: TF.to_tensor(img) * 2.0 - 1.0

        return {
            self.source_key: to_tensor(images[0]),
            self.target_key: to_tensor(images[1]),
            self.reflection_key: to_tensor(images[2]),
            "has_gt_reflection": torch.tensor(1.0),
            "filename": fn,
        }


class SIRSTestDataset(Dataset):
    """SIRS test/validation dataset.

    No augmentation. Images are aligned to a multiple of align_size
    (or resized to fixed_size if set). Loads R from reflection_layer/
    if available, otherwise computes R = I - T.
    """

    def __init__(self, data_dir, align_size=32, fixed_size=None, source_key="source_image",
                 target_key="target_image", reflection_key="reflection_image"):
        self.data_dir = data_dir
        self.align_size = align_size
        self.fixed_size = fixed_size
        self.source_key = source_key
        self.target_key = target_key
        self.reflection_key = reflection_key

        self.blended_dir = os.path.join(data_dir, "blended")
        self.trans_dir = os.path.join(data_dir, "transmission_layer")
        self.refl_dir = os.path.join(data_dir, "reflection_layer")
        self.has_refl_dir = os.path.isdir(self.refl_dir)

        self.filenames = sorted([
            f for f in os.listdir(self.blended_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        refl_src = "reflection_layer/" if self.has_refl_dir else "I - T (computed)"
        print(f"[SIRSTestDataset] {len(self.filenames)} images from {data_dir}  (R source: {refl_src})")

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

        blended = Image.open(os.path.join(self.blended_dir, fn)).convert('RGB')
        trans = Image.open(os.path.join(self.trans_dir, fn)).convert('RGB')

        blended = self._align(blended)
        if trans.size != blended.size:
            trans = trans.resize(blended.size, Image.BICUBIC)

        to_tensor = lambda img: TF.to_tensor(img) * 2.0 - 1.0

        refl_path = os.path.join(self.refl_dir, fn) if self.has_refl_dir else None
        if refl_path is not None and os.path.isfile(refl_path):
            refl = Image.open(refl_path).convert('RGB')
            refl = self._align(refl)
            if refl.size != blended.size:
                refl = refl.resize(blended.size, Image.BICUBIC)
            refl_tensor = to_tensor(refl)
        else:
            refl_tensor = to_tensor(blended) - to_tensor(trans)

        return {
            self.source_key: to_tensor(blended),
            self.target_key: to_tensor(trans),
            self.reflection_key: refl_tensor,
            "filename": fn,
        }


class SynthesisDataset(Dataset):
    """On-the-fly reflection synthesis dataset (DSRNet-style).

    Loads images from a single directory, splits them into transmission and
    reflection pools, and synthesizes blended images at runtime.
    """

    def __init__(self, data_dir, patch_size=512, source_key="source_image",
                 target_key="target_image", reflection_key="reflection_image",
                 synthesis_method="rdnet"):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.source_key = source_key
        self.target_key = target_key
        self.reflection_key = reflection_key

        exts = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

        if os.path.isdir(data_dir):
            all_files = []
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if os.path.splitext(f)[1] in exts:
                        all_files.append(os.path.join(root, f))
            self.paths = sorted(all_files)
        else:
            raise ValueError(f"Data directory not found: {data_dir}")

        if len(self.paths) < 2:
            raise ValueError(
                f"SynthesisDataset requires at least 2 images, found {len(self.paths)} in {data_dir}"
            )

        if synthesis_method == "physical":
            from .synthesis import PhysicalReflectionSynthesis
            self.syn_model = PhysicalReflectionSynthesis()
        else:
            from .synthesis import RDNetReflectionSynthesis
            self.syn_model = RDNetReflectionSynthesis()

        self.reset(shuffle=False)

        print(f"[SynthesisDataset] {len(self.paths)} images from {data_dir}")
        print(f"  Split: {len(self.trans_paths)} transmission, {len(self.refl_paths)} reflection")

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2
        self.trans_paths = self.paths[0:num_paths]
        self.refl_paths = self.paths[num_paths:2 * num_paths]

    def data_synthesis(self, trans_img, refl_img):
        trans_img, refl_img = paired_data_transforms(
            trans_img, refl_img, unaligned_transforms=False, patch_size=self.patch_size,
        )
        trans_np, refl_np, blended_np = self.syn_model(trans_img, refl_img)

        trans_t = TF.to_tensor(trans_np) * 2.0 - 1.0
        refl_t = TF.to_tensor(refl_np) * 2.0 - 1.0
        blended_t = TF.to_tensor(blended_np) * 2.0 - 1.0

        return trans_t, refl_t, blended_t

    def __len__(self):
        return max(len(self.trans_paths), len(self.refl_paths))

    def __getitem__(self, idx):
        if len(self.trans_paths) == 0 or len(self.refl_paths) == 0:
            raise RuntimeError(
                f"Dataset has no images: trans={len(self.trans_paths)}, refl={len(self.refl_paths)}"
            )

        trans_idx = idx % len(self.trans_paths)
        refl_idx = idx % len(self.refl_paths)

        trans_path = self.trans_paths[trans_idx]
        refl_path = self.refl_paths[refl_idx]

        trans_img = Image.open(trans_path).convert('RGB')
        refl_img = Image.open(refl_path).convert('RGB')

        trans_t, refl_t, blended_t = self.data_synthesis(trans_img, refl_img)

        return {
            self.source_key: blended_t,
            self.target_key: trans_t,
            self.reflection_key: refl_t,
            "has_gt_reflection": torch.tensor(1.0),
            "filename": os.path.basename(trans_path),
        }
