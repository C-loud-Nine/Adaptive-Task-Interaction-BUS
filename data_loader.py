"""Data loading and preprocessing for BUSI dataset."""

import os
import glob
import logging
import cv2
import numpy as np
import albumentations as A
from typing import Tuple
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
    """Apply config.INPUT_SCALE ("raw": [0, 255], the default; "unit": [0, 1])."""
    image = image.astype(np.float32)
    return image / 255.0 if config.INPUT_SCALE == "unit" else image


def load_whu_dataset(dataset_path: str):
    """Load BUSI-WHU v3: one folder per class, mask path from WHU_MASK_PATTERN.

    Returns images, masks, labels and source paths, in sorted file order.
    """
    items = []
    for ci, cls in enumerate(config.WHU_CLASSES):
        folder = os.path.join(dataset_path, cls)
        for image_path in sorted(glob.glob(os.path.join(folder, config.WHU_IMAGE_GLOB))):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            if stem.endswith("_mask"):
                continue
            mask_path = os.path.join(folder, config.WHU_MASK_PATTERN.format(stem=stem))
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"No mask for {image_path}; set WHU_MASK_PATTERN")
            items.append((ci, cls, image_path, mask_path))

    def load_one(item):
        ci, cls, image_path, mask_path = item
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = scale_image(cv2.resize(image, (224, 224)))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        return image, np.expand_dims((mask > 127).astype(np.float32), -1), ci, os.path.join(cls, os.path.basename(image_path))

    # files are read in parallel (I/O bound); results keep the sorted file order
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as pool:
        out = list(pool.map(load_one, items))
    images, masks, labels, paths = (list(x) for x in zip(*out)) if out else ([], [], [], [])
    return np.array(images), np.array(masks), np.array(labels), np.array(paths)


def load_busi_dataset(dataset_path: str, return_paths: bool = False):
    """
    Load BUSI dataset with images, masks, and labels.

    Files are read in sorted order so that a given random_state always
    produces the same partition on any machine.

    Returns:
        images: (N, 224, 224, 3) float32, scaled by config.INPUT_SCALE
        masks: (N, 224, 224, 1) float32 binary
        labels: (N,) int32
        paths: (N,) source file names, only if return_paths is True
    """
    images, masks, class_labels, paths = [], [], [], []

    def load_one(item):
        category, file = item
        image_path = os.path.join(dataset_path, category, file)
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = scale_image(cv2.resize(image, (224, 224)))
            mask = load_image_and_merge_masks(image_path)
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims((mask > 127).astype(np.float32), axis=-1)
            if image.shape != (224, 224, 3) or mask.shape != (224, 224, 1):
                logger.warning(f"Invalid shape: {file}")
                return None
            return image, mask, config.CATEGORIES.index(category), os.path.join(category, file)
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            return None

    items = [(category, file) for category in config.CATEGORIES
             for file in sorted(os.listdir(os.path.join(dataset_path, category)))
             if file.endswith(".png") and "_mask" not in file]
    # files are read in parallel (I/O bound); results keep the sorted file order
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as pool:
        for r in pool.map(load_one, items):
            if r is not None:
                images.append(r[0]); masks.append(r[1]); class_labels.append(r[2]); paths.append(r[3])
    
    if return_paths:
        return np.array(images), np.array(masks), np.array(class_labels), np.array(paths)
    return np.array(images), np.array(masks), np.array(class_labels)


def augment_dataset(images: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate augmented versions of training data."""
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    
    # albumentations >= 1.4 ignores the global seeds, so the pipeline is seeded
    # from NumPy, which train.py seeds with the training seed.
    import inspect
    seed_kw = ({"seed": int(np.random.randint(0, 2**31 - 1))}
               if "seed" in inspect.signature(A.Compose.__init__).parameters else {})
    transform = A.Compose([
        A.HorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        # border mode fixed explicitly: the default differs between
        # albumentations 1.x (reflect) and 2.x (constant)
        A.Rotate(limit=config.ROTATION_LIMIT, p=config.ROTATION_PROB,
                 border_mode=cv2.BORDER_REFLECT_101),
    ], **seed_kw)
    
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


def _union_find(n):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return find, union


def duplicate_groups(images: np.ndarray, paths: np.ndarray, dataset: str):
    """Group duplicate and near-duplicate images.

    Two sources are merged with union-find:
      * BUSI only: the published duplicate list (config.DUPLICATE_LIST);
      * both datasets: 64x64 grayscale thumbnails whose Pearson correlation is
        at least config.NEAR_DUP_CORR.
    Returns (group_id per image, report dict).
    """
    n = len(images)
    find, union = _union_find(n)
    report = {"list_pairs": 0, "auto_pairs": 0}

    if dataset == "busi" and os.path.exists(config.DUPLICATE_LIST):
        name_to_idx = {os.path.basename(str(p)): i for i, p in enumerate(paths)}
        members = {}
        with open(config.DUPLICATE_LIST) as fh:
            next(fh)
            for line in fh:
                g, f = line.rstrip("\n").split(",", 1)
                if f in name_to_idx:
                    members.setdefault(g, []).append(name_to_idx[f])
        for idx in members.values():
            for j in idx[1:]:
                union(idx[0], j)
                report["list_pairs"] += 1
        multi = [v for v in members.values() if len(v) > 1]
        report["list_groups"] = len(multi)
        report["list_images"] = sum(len(v) for v in multi)
        report["list_member"] = {i for v in multi for i in v}

    s = config.NEAR_DUP_SIZE
    thumbs = np.stack([cv2.resize(cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY),
                                  (s, s), interpolation=cv2.INTER_AREA).ravel()
                       for img in images]).astype(np.float64)
    thumbs -= thumbs.mean(axis=1, keepdims=True)
    thumbs /= np.maximum(np.linalg.norm(thumbs, axis=1, keepdims=True), 1e-12)
    corr = thumbs @ thumbs.T
    ii, jj = np.where(np.triu(corr, k=1) >= config.NEAR_DUP_CORR)
    auto = set()
    roots_before = [find(i) for i in range(n)]
    for i, j in zip(ii, jj):
        auto.add((int(i), int(j)))
    # auto pairs already linked by the published list (agreement check)
    report["auto_pairs_in_list"] = sum(roots_before[i] == roots_before[j] for i, j in auto)
    for i, j in auto:
        union(i, j)
    report["auto_pairs"] = len(auto)

    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    gid = np.array([remap[r] for r in roots])
    sizes = np.bincount(gid)
    big = np.argsort(sizes)[::-1][:5]
    report["largest_groups"] = [[str(paths[i]) for i in np.where(gid == g)[0][:6]] for g in big if sizes[g] > 2]
    report.update({"n_groups": int(len(sizes)),
                   "images_in_multi_groups": int(sizes[sizes > 1].sum()),
                   "multi_groups": int((sizes > 1).sum()),
                   "largest_group": int(sizes.max())})
    return gid, report


def deduplicate(groups: np.ndarray, labels: np.ndarray, paths: np.ndarray, list_member=frozenset(),
                remove_duplicates=True):
    """Curate duplicate groups.

    Groups whose members carry different class labels are dropped entirely
    (the label is ambiguous). With remove_duplicates (DEDUP = "remove") only
    the member with the lowest file index of every other group is kept; without
    it (DEDUP = "group") all their members are kept and make_partition places
    each group in one partition. Returns (boolean keep mask, list of
    (index, reason, kept file)).
    """
    import re
    def order(i):
        # natural order: class folder, then the numbers in the file name
        parts = re.split(r"(\d+)", os.path.basename(str(paths[i])))
        return (str(paths[i]).split(os.sep)[0],
                [int(t) if t.isdigit() else t for t in parts])
    keep = np.ones(len(groups), dtype=bool)
    excluded = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        src = "published list" if any(i in list_member for i in idx) else "near-duplicate check"
        if len(set(labels[idx])) > 1:
            for i in idx:
                keep[i] = False
                excluded.append((int(i), f"conflicting class labels in duplicate group ({src})", ""))
            continue
        if not remove_duplicates:
            continue
        idx = sorted(idx, key=order)
        for i in idx[1:]:
            keep[i] = False
            excluded.append((int(i), f"duplicate ({src})", str(paths[idx[0]])))
    return keep, excluded


def make_partition(labels: np.ndarray, groups: np.ndarray):
    """Class-stratified 70/15/15 partition that keeps every group in one split.

    StratifiedGroupKFold with config.SPLIT_FOLDS folds: the first 70% of folds
    form the training set, the next 15% validation and the last 15% test.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    k = config.SPLIT_FOLDS
    n_val = int(round(k * config.VAL_SPLIT))
    n_test = int(round(k * config.TEST_SPLIT))
    fold = np.full(len(labels), -1)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)
    for f, (_, te) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        fold[te] = f
    assert (fold >= 0).all()
    split = np.where(fold < k - n_val - n_test, "train",
                     np.where(fold < k - n_test, "val", "test"))
    for g in np.unique(groups):
        assert len(set(split[groups == g])) == 1, f"group {g} crosses partitions"
    return split


def load_curated(dataset_path: str, dataset: str = "busi", write_log: bool = True):
    """Load a dataset and apply the curation steps described in the README
    (mask-label check, duplicate grouping and removal).

    Returns images, masks, labels, paths, groups, categories and the report;
    the excluded files are written to results/excluded_<dataset>.csv."""
    logger.info(f"Loading {dataset} dataset...")
    if dataset == "busi_whu":
        images, masks, labels, paths = load_whu_dataset(dataset_path)
        categories = config.WHU_CLASSES
    else:
        images, masks, labels, paths = load_busi_dataset(dataset_path, return_paths=True)
        categories = config.CATEGORIES

    logger.info(f"Loaded {len(images)} samples")
    logger.info(f"Class distribution: {np.bincount(labels)}")

    # Mask-label consistency: lesion classes need a non-empty mask, the normal
    # class an empty one. Inconsistent images are excluded and logged.
    has_fg = masks.reshape(len(masks), -1).max(axis=1) > 0
    lesion = np.array([categories[int(c)] != "normal" for c in labels])
    ok = has_fg == lesion
    log_rows = [(str(paths[i]), categories[int(labels[i])], "mask inconsistent with class label", "")
                for i in np.where(~ok)[0]]
    images, masks, labels, paths = images[ok], masks[ok], labels[ok], paths[ok]

    groups, report = duplicate_groups(images, paths, dataset)
    report["invalid_masks"] = len(log_rows)
    if config.DEDUP in ("remove", "group"):
        keep, excluded = deduplicate(groups, labels, paths, report.get("list_member", set()),
                                     remove_duplicates=config.DEDUP == "remove")
        log_rows += [(str(paths[i]), categories[int(labels[i])], reason, rep)
                     for i, reason, rep in excluded]
        report["excluded_duplicates"] = len(excluded)
        images, masks, labels, paths, groups = (images[keep], masks[keep], labels[keep],
                                                paths[keep], groups[keep])
    reasons = sorted({r[2] for r in log_rows})
    report["excluded_by_reason"] = {r: sum(1 for x in log_rows if x[2] == r) for r in reasons}
    if write_log:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        import csv
        with open(os.path.join(config.RESULTS_DIR, f"excluded_{dataset}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "class", "reason", "kept_representative"])
            w.writerows(log_rows)
    report.pop("list_member", None)
    report["n_images_used"] = int(len(images))
    report["class_counts_used"] = {categories[c]: int((labels == c).sum()) for c in range(len(categories))}
    logger.info(f"Duplicate handling ({config.DEDUP}): {report}")
    return images, masks, labels, paths, groups, categories, report


def _thumbs(images):
    s = config.NEAR_DUP_SIZE
    t = np.stack([cv2.resize(cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY),
                             (s, s), interpolation=cv2.INTER_AREA).ravel() for img in images]).astype(np.float64)
    t -= t.mean(axis=1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return t


def external_busi_lesions(busi_path: str, whu_path: str = None):
    """External test set for models trained on BUSI-WHU: every curated BUSI
    image of a lesion class (benign, malignant), labelled with the BUSI-WHU
    class indices (benign 0, malignant 1). If whu_path is given, BUSI images
    that are near-exact copies (thumbnail correlation >= NEAR_DUP_CORR) of any
    BUSI-WHU image are removed, so no external test image was seen in
    training; the number removed is returned as "n_cross_dataset_duplicates"."""
    images, masks, labels, paths, _, categories, report = load_curated(busi_path, "busi", write_log=False)
    keep = np.array([categories[int(c)] != "normal" for c in labels])
    n_cross = 0
    if whu_path is not None:
        w_imgs = load_whu_dataset(whu_path)[0]
        corr = _thumbs(images) @ _thumbs(w_imgs).T
        cross = corr.max(axis=1) >= config.NEAR_DUP_CORR
        n_cross = int((cross & keep).sum())
        keep &= ~cross
    logger.info(f"External BUSI lesion set: {int(keep.sum())} images; "
                f"{n_cross} removed as near-exact copies of BUSI-WHU images")
    remap = {categories.index("benign"): config.WHU_CLASSES.index("benign"),
             categories.index("malignant"): config.WHU_CLASSES.index("malignant")}
    y = np.array([remap[int(c)] for c in labels[keep]])
    m = masks[keep]
    if m.ndim == 3:
        m = np.expand_dims(m, -1)
    return {"X": images[keep], "y": y, "mask": m, "paths": paths[keep],
            "n_cross_dataset_duplicates": n_cross}


def prepare_datasets(dataset_path: str, dataset: str = "busi", augment: bool = True) -> dict:
    """
    Load, curate and split a dataset.

    Returns a dict with X/y/mask for train (augmented), val and test, the
    un-augmented training images (X_train_orig, used to calibrate AIW), the
    file paths of each partition and the curation report.
    """
    images, masks, labels, paths, groups, categories, report = load_curated(dataset_path, dataset)
    split = make_partition(labels, groups)

    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    X_train, y_train, mask_train, p_train = images[tr], labels[tr], masks[tr], paths[tr]
    X_val, y_val, mask_val, p_val = images[va], labels[va], masks[va], paths[va]
    X_test, y_test, mask_test, p_test = images[te], labels[te], masks[te], paths[te]

    # Partition manifest (file, class, split, group) for traceability.
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    manifest = os.path.join(config.RESULTS_DIR, f"split_manifest_{dataset}.csv")
    import csv, hashlib
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "split", "group"])
        for i in range(len(images)):
            w.writerow([paths[i], categories[int(labels[i])], split[i], groups[i]])
    report["manifest_sha256"] = hashlib.sha256(open(manifest, "rb").read()).hexdigest()
    report["split_sizes"] = {s_: int((split == s_).sum()) for s_ in ("train", "val", "test")}
    logger.info(f"Split manifest written to {manifest} (sha256 {report['manifest_sha256']})")

    X_train_orig = X_train.copy()
    if augment:
        logger.info("Augmenting training data...")
        X_train, mask_train = augment_dataset(X_train, mask_train)
        y_train = np.tile(y_train, config.NUM_AUGMENTATIONS + 1)
        idx = np.random.permutation(len(X_train))
        X_train, mask_train, y_train = X_train[idx], mask_train[idx], y_train[idx]

    if mask_train.ndim == 3:
        mask_train = np.expand_dims(mask_train, -1)
    if mask_val.ndim == 3:
        mask_val = np.expand_dims(mask_val, -1)
    if mask_test.ndim == 3:
        mask_test = np.expand_dims(mask_test, -1)

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Val labels: {np.bincount(y_val)} | Test labels: {np.bincount(y_test)}")

    return {
        'X_train': X_train, 'y_train': y_train, 'mask_train': mask_train,
        'X_val': X_val, 'y_val': y_val, 'mask_val': mask_val,
        'X_test': X_test, 'y_test': y_test, 'mask_test': mask_test,
        'X_train_orig': X_train_orig,
        'paths_train': p_train, 'paths_val': p_val, 'paths_test': p_test,
        'dedup_report': report,
    }
