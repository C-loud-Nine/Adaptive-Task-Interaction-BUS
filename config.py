"""Configuration for the adaptive bidirectional task interaction model."""

import os

# ============================================================================
# PATHS
# ============================================================================
BUSI_PATH = './data/BUSI'
WHU_PATH = './data/BUSI-WHU'
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# DATASET SELECTION
# ============================================================================
# 'busi'     : 3 classes (normal, benign, malignant), masks may be split across files
# 'busi_whu' : 2 classes (benign, malignant)
DATASET = 'busi'

BUSI_CATEGORIES = ['normal', 'benign', 'malignant']
WHU_CATEGORIES = ['benign', 'malignant']

CATEGORIES = BUSI_CATEGORIES if DATASET == 'busi' else WHU_CATEGORIES
NUM_CLF_CLASSES = len(CATEGORIES)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
INPUT_SIZE = (224, 224, 3)
DECODER_CHANNELS = [384, 192, 96, 48]
NUM_SEG_CLASSES = 1
DROPOUT_RATE = 0.3
FREEZE_LAYERS = 50

# TIM modulation. Both factors are sigmoid-bounded, so mu lies in [1, 1+TAU]:
# the modulation can amplify but not attenuate.
TAU = 0.7

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 24
VAL_BATCH_SIZE = 24
EPOCHS = 100
WARMUP_EPOCHS = 3

INITIAL_LR = 3.0e-4
MIN_LR = 1.5e-6
COSINE_ALPHA = 0.03

ADAM_BETA_1 = 0.91
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-07
GLOBAL_CLIPNORM = 1.0

# ============================================================================
# DATA SPLIT
# ============================================================================
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
EARLY_STOPPING_PATIENCE = 10   # matches the value reported in the paper
EARLY_STOPPING_MIN_DELTA = 0.004
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.55
REDUCE_LR_COOLDOWN = 2
MONITOR = "val_combined"
MODE = "max"

# ============================================================================
# EVALUATION
# ============================================================================
# Threshold applied to the sigmoid segmentation output before computing
# foreground IoU and Dice. See evaluate.py.
SEG_THRESHOLD = 0.5

# ============================================================================
# AUGMENTATION
# ============================================================================
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ROTATION_LIMIT = 15
ROTATION_PROB = 0.7
NUM_AUGMENTATIONS = 3
