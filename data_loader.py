"""
Data loading and preprocessing for BUSI and BUSI-WHU.

Two changes from the earlier release:

1. BUSI-WHU is now supported. The previous version was BUSI-only, so the
   BUSI-WHU results reported in the paper could not be reproduced from it.

2. The split now reads TRAIN/VAL/TEST_SPLIT from config.py. The previous
   loader declared 70/15/15 in config but hard-coded 0.40 then 0.625, giving
   60/15/25. The constants are now honoured, so the code matches the protocol
   stated in the paper.

The shuffle after augmentation is also seeded, which it previously was not.
"""

import os
import glob
import logging
from typing import Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)


def load_image_and_merge_masks(image_path: str,
                               expected_shape: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Merge multiple mask files for one image using a pixel-wise maximum.

    BUSI stores some lesions as several `_mask`, `_mask_1`, ... files.
    """
    stem, ext = os.path.splitext(image_path)
    mask_files = sorted(glob.glob(f"{stem}_mask*{ext}"))
    if not mask_files:
        raise FileNotFoundError(f"No masks found for {image_path}")

    merged_mask = None
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, expected_shape, interpolation=cv2.INTER_NEAREST)
        merged_mask = mask if merged_mask is None else np.maximum(merged_mask, mask)

    if merged_mask is None:
        raise ValueError(f"No valid masks for {image_path}")
    return merged_mask


def _load_single_mask(image_path: str, mask_dir: str = None) -> np.ndarray:
    """Load one mask for BUSI-WHU, trying the common naming conventions."""
    folder = mask_dir or os.path.dirname(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    for candidate in (f"{stem}{ext}", f"{stem}.png", f"{stem}_mask{ext}", f"{stem}_mask.png"):
        path = os.path.join(folder, candidate)
        if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(image_path):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    raise FileNotFoundError(f"No mask found for {image_path}")


def load_busi_dataset(dataset_path: str):
    """BUSI: three class folders, masks alongside the images, possibly several per case."""
    images, masks, class_labels = [], [], []

    for category in config.BUSI_CATEGORIES:
        category_folder = os.path.join(dataset_path, category)
        if not os.path.isdir(category_folder):
            raise FileNotFoundError(f"Missing BUSI category folder: {category_folder}")

        for file in sorted(os.listdir(category_folder)):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')) or "_mask" in file:
                continue
            image_path = os.path.join(category_folder, file)
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMG_SIZE).astype(np.float32) / 255.0

                mask = load_image_and_merge_masks(image_path)
                mask = (mask > 127).astype(np.float32)[..., None]

                if image.shape != (224, 224, 3) or mask.shape != (224, 224, 1):
                    logger.warning(f"Invalid shape, skipping: {file}")
                    continue

                images.append(image)
                masks.append(mask)
                class_labels.append(config.BUSI_CATEGORIES.index(category))
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue

    return np.array(images), np.array(masks), np.array(class_labels)


def load_busi_whu_dataset(dataset_path: str):
    """BUSI-WHU: two class folders, one mask per image.

    Accepts either `<class>/images` and `<class>/masks` subfolders, or a flat
    `<class>` folder with `*_mask` files alongside the images.
    """
    images, masks, class_labels = [], [], []

    for category in config.WHU_CATEGORIES:
        base = os.path.join(dataset_path, category)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Missing BUSI-WHU category folder: {base}")

        img_dir = os.path.join(base, 'images')
        msk_dir = os.path.join(base, 'masks')
        if not os.path.isdir(img_dir):
            img_dir, msk_dir = base, base

        for file in sorted(os.listdir(img_dir)):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')) or "_mask" in file:
                continue
            image_path = os.path.join(img_dir, file)
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMG_SIZE).astype(np.float32) / 255.0

                mask = _load_single_mask(image_path, msk_dir)
                mask = (mask > 127).astype(np.float32)[..., None]

                images.append(image)
                masks.append(mask)
                class_labels.append(config.WHU_CATEGORIES.index(category))
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue

    return np.array(images), np.array(masks), np.array(class_labels)


def augment_dataset(images: np.ndarray, masks: np.ndarray):
    """Offline augmentation of the training split only: flips and rotation."""
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]

    transform = A.Compose([
        A.HorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        A.VerticalFlip(p=config.VERTICAL_FLIP_PROB),
        A.Rotate(limit=config.ROTATION_LIMIT, p=config.ROTATION_PROB),
    ])

    augmented_images, augmented_masks = [images], [masks]
    for _ in range(config.NUM_AUGMENTATIONS):
        aug_imgs, aug_msks = [], []
        for img, msk in zip(images, masks):
            out = transform(image=img, mask=msk)
            aug_imgs.append(out['image'])
            aug_msks.append(out['mask'])
        augmented_images.append(np.array(aug_imgs))
        augmented_masks.append(np.array(aug_msks))

    return np.concatenate(augmented_images), np.concatenate(augmented_masks)


def prepare_datasets(dataset_path: str = None, dataset: str = None) -> dict:
    """Load one dataset and split it according to config.

    dataset: 'busi' or 'busi_whu'. Defaults to config.DATASET.
    """
    dataset = dataset or config.DATASET
    if dataset == 'busi':
        dataset_path = dataset_path or config.BUSI_PATH
        logger.info("Loading BUSI...")
        images, masks, labels = load_busi_dataset(dataset_path)
    elif dataset == 'busi_whu':
        dataset_path = dataset_path or config.WHU_PATH
        logger.info("Loading BUSI-WHU...")
        images, masks, labels = load_busi_whu_dataset(dataset_path)
    else:
        raise ValueError(f"Unknown dataset '{dataset}', expected 'busi' or 'busi_whu'")

    logger.info(f"Loaded {len(images)} samples")
    logger.info(f"Class distribution: {np.bincount(labels)}")

    # Split using the ratios declared in config.
    test_plus_val = config.VAL_SPLIT + config.TEST_SPLIT
    X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(
        images, labels, masks,
        test_size=test_plus_val,
        stratify=labels,
        random_state=config.RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(
        X_temp, y_temp, mask_temp,
        test_size=config.TEST_SPLIT / test_plus_val,
        stratify=y_temp,
        random_state=config.RANDOM_STATE,
    )

    logger.info("Augmenting training data...")
    X_train, mask_train = augment_dataset(X_train, mask_train)
    y_train = np.tile(y_train, config.NUM_AUGMENTATIONS + 1)

    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.permutation(len(X_train))
    X_train, mask_train, y_train = X_train[idx], mask_train[idx], y_train[idx]

    if mask_train.ndim == 3:
        mask_train = np.expand_dims(mask_train, -1)
    if mask_val.ndim == 3:
        mask_val = np.expand_dims(mask_val, -1)
    if mask_test.ndim == 3:
        mask_test = np.expand_dims(mask_test, -1)

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Split ratios: {config.TRAIN_SPLIT}/{config.VAL_SPLIT}/{config.TEST_SPLIT}")

    return {
        'X_train': X_train, 'y_train': y_train, 'mask_train': mask_train,
        'X_val': X_val, 'y_val': y_val, 'mask_val': mask_val,
        'X_test': X_test, 'y_test': y_test, 'mask_test': mask_test,
    }
