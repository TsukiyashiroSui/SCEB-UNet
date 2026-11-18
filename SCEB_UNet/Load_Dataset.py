"""Strict image/mask loading for ACC-UNet segmentation experiments."""

from pathlib import Path
import random

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp"}


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    return np.flip(image, axis=axis).copy(), np.flip(label, axis=axis).copy()


def random_rotate(image, label, corrected_interpolation=False):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(
        image,
        angle,
        order=1 if corrected_interpolation else 0,
        reshape=False,
        mode="constant",
    )
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant")
    return image.copy(), label.copy()


def _resize_pair(image, label, output_size, corrected_interpolation=False):
    height, width = output_size
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    if label.shape[:2] != (height, width):
        label = cv2.resize(
            label,
            (width, height),
            interpolation=(
                cv2.INTER_NEAREST if corrected_interpolation else cv2.INTER_LINEAR
            ),
        )
    return image, label


def _to_tensors(image, label):
    image_tensor = F.to_tensor(np.ascontiguousarray(image))
    label_array = np.ascontiguousarray((label > 0).astype(np.uint8))
    label_tensor = torch.from_numpy(label_array).long().unsqueeze(0)
    return {"image": image_tensor, "label": label_tensor}


class RandomGenerator:
    def __init__(self, output_size, preprocessing_version="legacy_v1"):
        self.output_size = tuple(output_size)
        if preprocessing_version not in {"legacy_v1", "corrected_v2"}:
            raise ValueError(f"Unsupported preprocessing version: {preprocessing_version}")
        self.corrected_interpolation = preprocessing_version == "corrected_v2"

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(
                image, label, corrected_interpolation=self.corrected_interpolation
            )
        image, label = _resize_pair(
            image,
            label,
            self.output_size,
            corrected_interpolation=self.corrected_interpolation,
        )
        return _to_tensors(image, label)


class ValGenerator:
    def __init__(self, output_size, preprocessing_version="legacy_v1"):
        self.output_size = tuple(output_size)
        if preprocessing_version not in {"legacy_v1", "corrected_v2"}:
            raise ValueError(f"Unsupported preprocessing version: {preprocessing_version}")
        self.corrected_interpolation = preprocessing_version == "corrected_v2"

    def __call__(self, sample):
        image, label = _resize_pair(
            sample["image"],
            sample["label"],
            self.output_size,
            corrected_interpolation=self.corrected_interpolation,
        )
        return _to_tensors(image, label)


def to_long_tensor(pic):
    """Compatibility helper retained for external scripts."""
    return torch.from_numpy(np.asarray(pic, dtype=np.uint8)).long()


def correct_dims(*images):
    corrected = [np.expand_dims(image, axis=2) if image.ndim == 2 else image for image in images]
    return corrected[0] if len(corrected) == 1 else tuple(corrected)


class ImageToImage2D(Dataset):
    """Load paired images and masks, failing fast on invalid data."""

    def __init__(
        self,
        dataset_path,
        joint_transform=None,
        one_hot_mask=False,
        image_size=224,
        n_labels=1,
    ):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.input_path = self.dataset_path / "img"
        self.output_path = self.dataset_path / "labelcol"
        self.one_hot_mask = one_hot_mask
        self.n_labels = n_labels
        self.joint_transform = joint_transform or ValGenerator((image_size, image_size))

        if not self.input_path.is_dir() or not self.output_path.is_dir():
            raise FileNotFoundError(
                f"Expected img/ and labelcol/ under dataset split: {self.dataset_path}"
            )

        image_paths = sorted(
            path for path in self.input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise RuntimeError(f"No images found in {self.input_path}")

        self.samples = [(image_path, self._find_masks(image_path)) for image_path in image_paths]
        print(f"Validated {len(self.samples)} image/mask pairs in {self.dataset_path}")

    def _find_masks(self, image_path):
        base = image_path.stem

        # BUSI can contain several masks for one image; union every exact mask variant.
        busi_masks = sorted(
            path for path in self.output_path.glob(f"{base}_mask*")
            if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS
        )
        if busi_masks:
            return tuple(busi_masks)

        candidates = [
            image_path.name,                    # Kvasir-SEG and exact-name layouts
            f"{base}_anno.bmp",                 # GlaS
            f"{base}_anno.png",
            f"{base}_segmentation.png",         # ISIC2018
            f"{base.replace('_training', '_manual1')}.gif",  # DRIVE
            f"{base.replace('_test', '_manual1')}.gif",
            f"{base}.png",
            f"{base}.bmp",
            f"{base}.gif",
        ]
        matches = [self.output_path / name for name in candidates if (self.output_path / name).is_file()]
        if not matches:
            raise FileNotFoundError(f"No exact mask found for {image_path.name} in {self.output_path}")
        return (matches[0],)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_paths = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        masks = []
        for mask_path in mask_paths:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Cannot read mask: {mask_path}")
            masks.append(mask)

        mask = np.maximum.reduce(masks)
        if self.n_labels == 1:
            mask = (mask > 0).astype(np.uint8)

        sample = self.joint_transform({"image": image, "label": mask})
        if self.one_hot_mask:
            classes = int(self.one_hot_mask)
            label = sample["label"]
            one_hot = torch.zeros((classes, label.shape[-2], label.shape[-1]))
            sample["label"] = one_hot.scatter_(0, label.long(), 1)

        return sample, image_path.name
