"""Configuration file for multi-task breast ultrasound model."""

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
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
INPUT_SIZE = (224, 224, 3)
DECODER_CHANNELS = [384, 192, 96, 48]
NUM_SEG_CLASSES = 1
NUM_CLF_CLASSES = 3
DROPOUT_RATE = 0.3

# AIW variant:
#   "adaptive": softmax over normalised activation dispersion (the paper's model)
#   "off":      no blend, the TIM output is used directly
#   "fixed":    fixed blend w_seg = w_clf = 0.5
#   "constant": two learned per-level constants (softmax, summing to 1)
AIW_MODE = "adaptive"

# Ablation switches (Table 2): MSCF in the residual blocks, TIM at the decoder levels
USE_MSCF = True
USE_TIM = True

# Input scaling in the loader. The Keras EfficientNet-B4 keeps its built-in
# rescaling and normalisation layers.
#   "unit": images scaled to [0, 1] (the paper's setting)
#   "raw":  images passed in [0, 255]
INPUT_SCALE = "unit"

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
# DATA SPLIT
# ============================================================================
# Stratified image-level 70/15/15 split (546/117/117 on BUSI, 648/139/140 on
# BUSI-WHU), written to results/split_manifest_{dataset}.csv.
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# ============================================================================
# LOSS WEIGHTS
# ============================================================================
# Total loss = SEG_WEIGHT * L_seg + CLF_WEIGHT * L_clf
SEG_WEIGHT = 0.82
CLF_WEIGHT = 0.18

# ============================================================================
# CALLBACKS
# ============================================================================
EARLY_STOPPING_PATIENCE = 22
EARLY_STOPPING_MIN_DELTA = 0.004
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.55
REDUCE_LR_COOLDOWN = 2
MONITOR = "val_combined"

# Segmentation term of val_combined (0.7 * IoU + 0.3 * accuracy).
#   "meaniou_raw": tf.keras.metrics.MeanIoU(num_classes=2) on the segmentation
#                  output (the paper's setting)
#   "binary_iou":  BinaryIoU on both classes after thresholding at 0.5
SELECTION_IOU = "meaniou_raw"
MODE = "max"

# ============================================================================
# AUGMENTATION
# ============================================================================
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ROTATION_LIMIT = 15
ROTATION_PROB = 0.7
NUM_AUGMENTATIONS = 3  # augmented copies per training image, plus the original

# ============================================================================
# CATEGORIES
# ============================================================================
CATEGORIES = ['normal', 'benign', 'malignant']
