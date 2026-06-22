"""Image transformation utilities for paired reflection-removal data."""

import math
import random
from PIL import Image
import torchvision.transforms.functional as TF


def scale_width(img, target_width):
    """Scale image to target width while maintaining aspect ratio. Even height."""
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.0) * 2
    return img.resize((w, h), Image.BICUBIC)


def scale_height(img, target_height):
    """Scale image to target height while maintaining aspect ratio. Even width."""
    ow, oh = img.size
    if oh == target_height:
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.0) * 2
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False, patch_size=224):
    """Apply identical random transforms to two PIL images.

    Pipeline: random resize -> random hflip -> random 90/180/270 rotation ->
    safe resize if too small -> random crop to patch_size x patch_size.
    """
    target_size = int(random.randint(patch_size, patch_size * 2) / 2.) * 2
    ow, oh = img_1.size

    if ow >= oh:
        img_1 = scale_height(img_1, target_size)
        img_2 = scale_height(img_2, target_size)
    else:
        img_1 = scale_width(img_1, target_size)
        img_2 = scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = TF.hflip(img_1)
        img_2 = TF.hflip(img_2)

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = TF.rotate(img_1, angle)
        img_2 = TF.rotate(img_2, angle)

    w, h = img_1.size
    if h < patch_size or w < patch_size:
        scale = max(patch_size / h, patch_size / w)
        new_h = math.ceil((int(h * scale) + 1) / 2.) * 2
        new_w = math.ceil((int(w * scale) + 1) / 2.) * 2
        img_1 = img_1.resize((new_w, new_h), Image.BICUBIC)
        img_2 = img_2.resize((new_w, new_h), Image.BICUBIC)

    w, h = img_1.size
    i = random.randint(0, h - patch_size)
    j = random.randint(0, w - patch_size)
    img_1 = TF.crop(img_1, i, j, patch_size, patch_size)

    if unaligned_transforms:
        i += random.randint(-10, 10)
        j += random.randint(-10, 10)

    img_2 = TF.crop(img_2, i, j, patch_size, patch_size)
    return img_1, img_2


def paired_transforms(images, patch_size=512):
    """Apply identical random transforms to a list of PIL images."""
    if not images:
        return images

    if len(images) == 2:
        img_1, img_2 = paired_data_transforms(
            images[0], images[1], unaligned_transforms=False, patch_size=patch_size,
        )
        return [img_1, img_2]

    target_size = int(random.randint(patch_size, patch_size * 2) / 2.) * 2
    ow, oh = images[0].size

    if ow >= oh:
        images = [scale_height(img, target_size) for img in images]
    else:
        images = [scale_width(img, target_size) for img in images]

    if random.random() < 0.5:
        images = [TF.hflip(img) for img in images]

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        images = [TF.rotate(img, angle) for img in images]

    w, h = images[0].size
    if h < patch_size or w < patch_size:
        scale = max(patch_size / h, patch_size / w)
        new_h = math.ceil((int(h * scale) + 1) / 2.) * 2
        new_w = math.ceil((int(w * scale) + 1) / 2.) * 2
        images = [img.resize((new_w, new_h), Image.BICUBIC) for img in images]

    w, h = images[0].size
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    images = [TF.crop(img, top, left, patch_size, patch_size) for img in images]

    return images
