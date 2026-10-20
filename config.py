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

# AIW variant (controls requested in the ISBI review, item V8-08):
#   "adaptive": softmax over batch-normalised dispersions (reported model)
#   "off":      no blend, the TIM output is used directly (w = 1)
#   "fixed":    fixed blend w_seg = w_clf = 0.5
#   "constant": two learned per-level constants (softmax, summing to 1), no input dependence
AIW_MODE = "adaptive"

# Input scaling before the Keras EfficientNet-B4, which contains its own
# Rescaling(1/255) and Normalization layers.
#   "unit": images divided by 255 in the loader, as in the reported runs, so
#           the encoder rescales a second time
#   "raw":  images passed in [0, 255], the range the Keras encoder expects
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
# data_loader.prepare_datasets() reads these three values: a stratified
# 70/15/15 split (546/117/117 on BUSI, 648/139/140 on BUSI-WHU), written to
# results/split_manifest_{dataset}.csv.
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# ============================================================================
# LOSS WEIGHTS
# ============================================================================
SEG_WEIGHT_START = 0.82
CLF_WEIGHT_START = 0.18
SEG_WEIGHT_FINAL = 0.80
CLF_WEIGHT_FINAL = 0.20

# ============================================================================
# CALLBACKS
# ============================================================================
EARLY_STOPPING_PATIENCE = 22
EARLY_STOPPING_MIN_DELTA = 0.004
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.55
REDUCE_LR_COOLDOWN = 2
MONITOR = "val_combined"

# Segmentation IoU used inside val_combined.
#   "meaniou_raw": tf.keras.metrics.MeanIoU(num_classes=2) on the sigmoid
#                  output, as in the reported runs. MeanIoU casts predictions
#                  to integer class labels, so probabilities below 1 count as
#                  background.
#   "binary_iou":  BinaryIoU on both classes after thresholding at 0.5.
SELECTION_IOU = "meaniou_raw"
MODE = "max"

# ============================================================================
# AUGMENTATION
# ============================================================================
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ROTATION_LIMIT = 15
ROTATION_PROB = 0.7
NUM_AUGMENTATIONS = 3  # Generate 3 augmented versions per sample

# ============================================================================
# CATEGORIES
# ============================================================================
CATEGORIES = ['normal', 'benign', 'malignant']
