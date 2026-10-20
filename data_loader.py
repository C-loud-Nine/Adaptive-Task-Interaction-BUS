"""Data loading and preprocessing for BUSI dataset."""

import os
import glob
import logging
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from typing import Tuple, List
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image_and_merge_masks(
    image_path: str,
    expected_shape: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Merge multiple mask files using pixel-wise maximum."""
    mask_pattern = image_path.replace(".png", "_mask*.png")
    mask_files = sorted(glob.glob(mask_pattern))
    
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


def scale_image(image: np.ndarray) -> np.ndarray:
    """Apply config.INPUT_SCALE ("unit" as in the reported runs, or "raw")."""
    image = image.astype(np.float32)
    return image / 255.0 if config.INPUT_SCALE == "unit" else image


def load_whu_dataset(dataset_path: str):
    """Load BUSI-WHU v3: one folder per class, mask path from WHU_MASK_PATTERN.

    Returns images, masks, labels and source paths, in sorted file order.
    """
    images, masks, labels, paths = [], [], [], []
    for ci, cls in enumerate(config.WHU_CLASSES):
        folder = os.path.join(dataset_path, cls)
        for image_path in sorted(glob.glob(os.path.join(folder, config.WHU_IMAGE_GLOB))):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            if stem.endswith("_mask"):
                continue
            mask_path = os.path.join(folder, config.WHU_MASK_PATTERN.format(stem=stem))
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"No mask for {image_path}; set WHU_MASK_PATTERN")
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = scale_image(cv2.resize(image, (224, 224)))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            masks.append(np.expand_dims((mask > 127).astype(np.float32), -1))
            images.append(image)
            labels.append(ci)
            paths.append(os.path.join(cls, os.path.basename(image_path)))
    return np.array(images), np.array(masks), np.array(labels), np.array(paths)


def load_busi_dataset(dataset_path: str, return_paths: bool = False):
    """
    Load BUSI dataset with images, masks, and labels.

    Files are read in sorted order so that a given random_state always
    produces the same partition on any machine.

    Returns:
        images: (N, 224, 224, 3) float32 [0, 1]
        masks: (N, 224, 224, 1) float32 binary
        labels: (N,) int32
        paths: (N,) source file names, only if return_paths is True
    """
    images, masks, class_labels, paths = [], [], [], []
    
    for category in config.CATEGORIES:
        category_folder = os.path.join(dataset_path, category)
        files = sorted(os.listdir(category_folder))
        
        for file in files:
            if not file.endswith(".png") or "_mask" in file:
                continue
            
            image_path = os.path.join(category_folder, file)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = scale_image(image)
                
                # Load and merge masks
                mask = load_image_and_merge_masks(image_path)
                mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)
                
                # Validate
                if image.shape != (224, 224, 3) or mask.shape != (224, 224, 1):
                    logger.warning(f"Invalid shape: {file}")
                    continue
                
                images.append(image)
                masks.append(mask)
                class_labels.append(config.CATEGORIES.index(category))
                paths.append(os.path.join(category, file))
                
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue
    
    if return_paths:
        return np.array(images), np.array(masks), np.array(class_labels), np.array(paths)
    return np.array(images), np.array(masks), np.array(class_labels)


def augment_dataset(images: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate augmented versions of training data."""
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    
    transform = A.Compose([
        A.HorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        A.VerticalFlip(p=config.VERTICAL_FLIP_PROB),
        A.Rotate(limit=config.ROTATION_LIMIT, p=config.ROTATION_PROB),
    ])
    
    augmented_images = [images]
    augmented_masks = [masks]
    
    for _ in range(config.NUM_AUGMENTATIONS):
        aug_imgs, aug_msks = [], []
        for img, msk in zip(images, masks):
            augmented = transform(image=img, mask=msk)
            aug_imgs.append(augmented['image'])
            aug_msks.append(augmented['mask'])
        augmented_images.append(np.array(aug_imgs))
        augmented_masks.append(np.array(aug_msks))
    
    return np.concatenate(augmented_images), np.concatenate(augmented_masks)


def prepare_datasets(dataset_path: str, dataset: str = "busi") -> dict:
    """
    Load and split BUSI dataset.
    
    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        mask_train, mask_val, mask_test
    """
    logger.info(f"Loading {dataset} dataset...")
    if dataset == "busi_whu":
        images, masks, labels, paths = load_whu_dataset(dataset_path)
        categories = config.WHU_CLASSES
    else:
        images, masks, labels, paths = load_busi_dataset(dataset_path, return_paths=True)
        categories = config.CATEGORIES
    
    logger.info(f"Loaded {len(images)} samples")
    logger.info(f"Class distribution: {np.bincount(labels)}")
    
    # Stratified train/validation/test split from config (0.70/0.15/0.15).
    # For 780 BUSI images this gives 546/117/117 before augmentation.
    holdout = config.VAL_SPLIT + config.TEST_SPLIT
    X_train, X_temp, y_train, y_temp, mask_train, mask_temp, p_train, p_temp = train_test_split(
        images, labels, masks, paths,
        test_size=holdout,
        stratify=labels,
        random_state=config.RANDOM_STATE
    )
    X_val, X_test, y_val, y_test, mask_val, mask_test, p_val, p_test = train_test_split(
        X_temp, y_temp, mask_temp, p_temp,
        test_size=config.TEST_SPLIT / holdout,
        stratify=y_temp,
        random_state=config.RANDOM_STATE
    )
    
    # Write the partition manifest so every reported number can be traced
    # to the exact image files it was computed on.
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    manifest = os.path.join(config.RESULTS_DIR, f"split_manifest_{dataset}.csv")
    with open(manifest, "w") as fh:
        fh.write("file,class,split\n")
        for part, ps, ys in (("train", p_train, y_train), ("val", p_val, y_val), ("test", p_test, y_test)):
            for f, y in zip(ps, ys):
                fh.write(f"{f},{categories[int(y)]},{part}\n")
    logger.info(f"Split manifest written to {manifest}")
    
    # Augment training data
    logger.info("Augmenting training data...")
    X_train, mask_train = augment_dataset(X_train, mask_train)
    y_train = np.tile(y_train, config.NUM_AUGMENTATIONS + 1)
    
    # Shuffle augmented training data
    idx = np.random.permutation(len(X_train))
    X_train, mask_train, y_train = X_train[idx], mask_train[idx], y_train[idx]
    
    # Expand mask dimensions if needed
    if mask_train.ndim == 3:
        mask_train = np.expand_dims(mask_train, -1)
    if mask_val.ndim == 3:
        mask_val = np.expand_dims(mask_val, -1)
    if mask_test.ndim == 3:
        mask_test = np.expand_dims(mask_test, -1)
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Train labels: {np.bincount(y_train)}")
    logger.info(f"Val labels: {np.bincount(y_val)}")
    logger.info(f"Test labels: {np.bincount(y_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train, 'mask_train': mask_train,
        'X_val': X_val, 'y_val': y_val, 'mask_val': mask_val,
        'X_test': X_test, 'y_test': y_test, 'mask_test': mask_test,
        'paths_train': p_train, 'paths_val': p_val, 'paths_test': p_test,
    }
