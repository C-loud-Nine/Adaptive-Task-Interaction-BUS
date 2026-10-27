"""All settings of the paper (Sec. 3.1). Ablation and control switches are command-line flags of train.py."""

import os

# ============================================================================
# PATHS
# ============================================================================
BUSI_PATH = './data/BUSI'
BUSI_WHU_PATH = './data/BUSI-WHU'

# BUSI-WHU v3 (doi:10.17632/k6cpmwybk3.3): one folder per class under
# BUSI_WHU_PATH; each image <stem><ext> has its mask at the path given by
# WHU_MASK_PATTERN. Set the two patterns to the layout of the downloaded
# archive before the first run; the loader stops if any mask is missing.
WHU_CLASSES = ['benign', 'malignant']
WHU_IMAGE_GLOB = '*.png'
WHU_MASK_PATTERN = '{stem}_mask.png'
CHECKPOINT_DIR = './checkpoints'   # train.py sets both to its run folder
RESULTS_DIR = './results'

# ============================================================================
# MODEL
# ============================================================================
DROPOUT_RATE = 0.3

# Images are loaded in [0, 255] ("raw"); the model applies the ImageNet
# normalisation of its EfficientNet-B4 weights. "unit" scales to [0, 1].
INPUT_SCALE = "raw"

# Mixed precision on CUDA (float16 compute with loss scaling; AIW, losses and
# outputs in float32), for the proposed model and every comparison method.
MIXED_PRECISION = True

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 24
VAL_BATCH_SIZE = 24
EPOCHS = 100
WARMUP_EPOCHS = 3

# Learning rate
INITIAL_LR = 3.0e-4
MIN_LR = 1.5e-6
COSINE_ALPHA = 0.03

# Optimizer
ADAM_BETA_1 = 0.91
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-07
GLOBAL_CLIPNORM = 1.0

# ============================================================================
# DATA CURATION AND SPLIT
# ============================================================================
# Duplicate handling (README, Data). Images are linked into
# duplicate groups by
#   (i) the published BUSI duplicate list (Pawlowska et al., Data in Brief 48,
#       109247, 2023; splits/busi_duplicate_groups.csv), BUSI only, and
#  (ii) an automatic near-duplicate check on both datasets: 64x64 grayscale
#       thumbnails with Pearson correlation >= NEAR_DUP_CORR (near-exact copies).
# Groups whose members carry different class labels are dropped (ambiguous
# label) in both modes.
# DEDUP = "group":  keep all other images and place each group wholly in one
#                   partition (used for the paper);
#         "remove": additionally keep only one image per group (the lowest
#                   file index).
# The remaining images are split 70/15/15, stratified by class, with every
# group inside one partition. The partition is written to
# results/split_manifest_{dataset}.csv and the excluded images, with the
# reason for each, to results/excluded_{dataset}.csv.
DEDUP = "group"
NEAR_DUP_CORR = 0.98
NEAR_DUP_SIZE = 64
SPLIT_FOLDS = 20                # 14 / 3 / 3 folds -> 70 / 15 / 15
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42
DUPLICATE_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "splits", "busi_duplicate_groups.csv")

# ============================================================================
# LOSS WEIGHTS
# ============================================================================
# Total loss = SEG_WEIGHT * L_seg + CLF_WEIGHT * L_clf
SEG_WEIGHT = 0.82
CLF_WEIGHT = 0.18

# ============================================================================
# MODEL SELECTION AND SCHEDULE
# ============================================================================
EARLY_STOPPING_PATIENCE = 22
EARLY_STOPPING_MIN_DELTA = 0.004
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.55
REDUCE_LR_COOLDOWN = 2
# Model selection (computed by train.py after every epoch from epoch 4, with
# the model in inference mode and AIW calibrated as at test time):
#   val_score = SELECTION_IOU_WEIGHT * J_val + (1 - SELECTION_IOU_WEIGHT) * A_val
# J_val: foreground IoU pooled over all validation pixels after thresholding the
# predicted mask at 0.5; A_val: validation classification accuracy.
# Segmentation-only comparison methods (baselines/train_baseline.py) use J_val alone.
SELECTION_IOU_WEIGHT = 0.7
SEG_THRESHOLD = 0.5

# Segmentation loss: per-image focal Tversky on lesion-positive images (size
# weighted) + pixel-wise binary cross-entropy on all images + edge term.
BCE_WEIGHT = 1.0
EDGE_WEIGHT = 0.15

# ============================================================================
# AUGMENTATION
# ============================================================================
HORIZONTAL_FLIP_PROB = 0.5
# No vertical flip: it would put the skin line at the bottom of the image and
# invert posterior acoustic shadowing / enhancement (a malignancy cue).
ROTATION_LIMIT = 15
ROTATION_PROB = 0.7
NUM_AUGMENTATIONS = 3  # augmented copies per training image, plus the original

# ============================================================================
# CATEGORIES
# ============================================================================
CATEGORIES = ['normal', 'benign', 'malignant']
