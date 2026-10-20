# Adaptive Bidirectional Task Interaction for Joint Segmentation and Classification of Breast Ultrasound

<div align="center">

[![Python 3.9-3.11](https://img.shields.io/badge/python-3.9--3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official implementation**

</div>

---

## Overview

Breast ultrasound interpretation requires lesion segmentation and tissue
classification at the same time. Most multi-task models for this setting share
an encoder, and some pass segmentation features to the classifier; once the two
tasks enter separate decoders they no longer exchange information. This
repository exchanges latent features in both directions at every decoder level
and blends original and interacted features with learned coefficients.

This repository implements a framework with three components:

* **TIM (Task Interaction Module)**, applied at all four decoder resolutions,
  giving bidirectional segmentation/classification exchange during decoding.
* **MSCF (Multi-Scale Context Fusion)**, parallel dilated separable
  convolutions with instance-adaptive scale weighting, to cover lesion size
  variation.
* **AIW (Adaptive Interaction Weighting)**, a per-level softmax gate that
  decides how much of each TIM-enhanced representation to adopt.

Reported results, baselines and ablations live in the paper. They are
deliberately **not** duplicated here: a second copy of a results table drifts
out of step with the manuscript. `evaluate.py` (below) regenerates the proposed
model's metrics from a checkpoint; the paper's Dice column equals
`max_mean_dice_implied_by_mean_iou_percent` in its summary. Baselines and the
MSCF/TIM ablation variants are not part of this release.

---

## Architecture

| Decoder level | Resolution | Channels | Interaction focus |
|:-------------:|:----------:|:--------:|:------------------|
| D1 | 14x14 | 384 | Coarse semantic context |
| D2 | 28x28 | 192 | Mid-level feature fusion |
| D3 | 56x56 | 96 | Fine-grained spatial detail |
| D4 | 112x112 | 48 | Boundary-level precision |

### Algorithm 1: Task Interaction Module

```
Input:  decoder features D_l in R^(B x H x W x C),
        classification features f_clf^l in R^(B x 256)
Output: enhanced pair (D_l^enh, f_clf^(l,enh))

# segmentation -> classification
1: f_seg^ctx    <- Dense_256(GAP(Conv_1x1(D_l)))
2: g_clf        <- sigmoid(MLP(f_seg^ctx))
3: f_clf^(l,enh) <- f_clf^l + g_clf * f_seg^ctx

# classification -> segmentation
4: f_clf^spat <- Reshape(Dense_C(f_clf^l), [1, 1, C])
5: g_seg      <- sigmoid(MLP(GAP(D_l)))
6: mu_l       <- 1 + 0.7 * g_seg * f_clf^spat        # in [1, 1.7], amplify-only
7: D_l^enh    <- D_l * mu_l
```

The modulation in step 6 is one-sided by construction: both factors are
sigmoids, so `mu_l` never falls below 1 and the pathway can amplify but not
suppress.

### Algorithm 2: Adaptive Interaction Weighting

```
Input:  pre-interaction (D_l, f_clf^l), enhanced (D_l^enh, f_clf^(l,enh))
Output: final pair (D_l^final, f_clf^(l,final))

1: s_seg  <- E_c[Var_(h,w)(D_l^enh)]        # activation dispersion, per channel
2: s_clf  <- Var(f_clf^(l,enh))             # activation dispersion of the vector
3: s~_seg <- s_seg / (mean_batch(s_seg) + eps)
4: s~_clf <- s_clf / (mean_batch(s_clf) + eps)
5: [w_seg, w_clf] <- Softmax(MLP_2(ReLU(MLP_32([s~_seg, s~_clf]))))
6: D_l^final      <- D_l + w_seg * (D_l^enh - D_l)
7: f_clf^(l,final) <- f_clf^l + w_clf * (f_clf^(l,enh) - f_clf^l)
```

Two properties of this gate are worth stating plainly, because they are easy
to misread:

* **The two signals are activation dispersion, not predictive uncertainty.**
  They come from a single deterministic forward pass, with no ensemble, no
  dropout sampling and no density model.
* **Steps 3 and 4 normalise over the current batch, at inference as well as in
  training.** The weights for one image therefore depend on the other images
  evaluated alongside it. At batch size 1 both normalised inputs collapse to
  approximately 1 and the gate reduces to a learned per-level constant. This
  is the implemented behaviour; `evaluate.py` records the evaluation batch size
  for that reason.

### Algorithm 3: Multi-Scale Context Fusion

```
Input:  X in R^(B x H x W x C)
Output: Y in R^(B x H x W x C)

1: F1 <- SepConv_3x3^(r=1)(X)            # 3x3 effective receptive field
2: F2 <- SepConv_3x3^(r=2)(X)            # 5x5 effective receptive field
3: F4 <- SepConv_3x3^(r=4)(X)            # 9x9 effective receptive field
4: a  <- Softmax(MLP_3(ReLU(MLP_(C/8)(GAP(X)))))
5: Y  <- Conv_1x1(sum_i a_i * F_i)
```

The softmax over scales makes the three branches compete rather than sum.

---

## Repository structure

```
.
|-- config.py           # hyperparameters, dataset paths, split ratios
|-- data_loader.py      # BUSI and BUSI-WHU loading, augmentation, splitting
|-- modules.py          # TaskInteractionModule, AdaptiveInteractionWeighting,
|                       # MultiScaleContextFusion, AttentionGate, ResidualBlock,
|                       # DualPathAttention
|-- model.py            # adaptive_interaction_model(), build_decoder_block(),
|                       # build_tim_aiw()
|-- loss.py             # segmentation + classification objectives
|-- train.py            # warmup + cosine training loop, composite monitoring
|-- evaluate.py         # per-image evaluation export (see below)
|-- batch_check.py      # predictions at several evaluation batch sizes
|-- audit_duplicates.py # exact and near-duplicate audit against the split
|-- complexity.py       # parameters and FLOPs
|-- requirements.txt
`-- README.md
```

---

## Dataset

**BUSI (Breast Ultrasound Images).** 780 images: 133 normal, 437 benign, 210
malignant, from 600 patients, variable resolution, expert annotated.
Source: <https://data.mendeley.com/datasets/wmy84gzngw/1>.
Reference: Al-Dhabyani et al., *Data in Brief* 28:104863, 2020.

Where an image has several mask files they are merged by pixel-wise maximum.
Normal cases carry an all-zero mask.

`data_loader.py` loads BUSI and BUSI-WHU v3 (`--dataset busi_whu` in
`train.py`, `evaluate.py` and `batch_check.py`); both use
the same stratified 70/15/15 split and write a partition manifest.

**Preprocessing**

* resize to 224x224, scale to [0, 1] (`config.INPUT_SCALE = "unit"`, as in the reported runs; the Keras EfficientNet-B4 then applies its own 1/255 rescaling, and `"raw"` passes [0, 255] instead);
* augmentation on the training split only: horizontal flip, vertical flip,
  rotation of +/- 15 degrees, three augmented copies per image plus the
  original;
* split stratified by **class label** (not by patient), `random_state = 42`.

**Split sizes.** `prepare_datasets()` reads `TRAIN_SPLIT / VAL_SPLIT /
TEST_SPLIT = 0.70 / 0.15 / 0.15` from `config.py`: 546/117/117 BUSI images
before augmentation. Files are read in sorted order and the partition is
written to `results/split_manifest_<dataset>.csv`.

---

## Training

```bash
pip install -r requirements.txt
python train.py
```

| Hyperparameter | Value (`config.py`; encoder and frozen layers in `model.py`) |
|:---------------|:-------------------------|
| Input resolution | 224 x 224 x 3 |
| Encoder | EfficientNet-B4, ImageNet, first 50 layers frozen |
| Decoder channels | 384 / 192 / 96 / 48 |
| Batch size (train / val) | 24 / 24 |
| Initial LR | 3.0e-4, cosine annealing (floor 0.03 x 3.0e-4) with plateau reduction down to 1.5e-6 |
| Optimizer | Adam, beta1 = 0.91, beta2 = 0.999, eps = 1e-7 |
| Gradient clipping | 1.0 global norm |
| Warmup epochs | 3 |
| Max epochs | 100, early stopping patience 22, min delta 0.004 |
| LR reduction | factor 0.55, patience 7, cooldown 2 |
| Selection metric | `val_combined` = 0.7 x IoU + 0.3 x accuracy, maximised. Reported runs: `MeanIoU(num_classes=2)` on the sigmoid output, which casts probabilities to class labels; `config.SELECTION_IOU = "binary_iou"` thresholds at 0.5 instead |
| Task weights | 0.82 / 0.18, set at compile; the callback in `train.py` does not change them in TF 2.13 (see `AdaptiveLossWeights`) |

### Loss

```
L_total = 0.82 * L_seg + 0.18 * L_clf

L_seg = L_focal_tversky + 0.25 * L_boundary + 0.15 * L_gradient_dispersion
L_clf = FocalCrossEntropy(gamma = 2.0)
```

Three facts about `loss.py` that the code settles:

1. `enhanced_lesion_focus_loss` calls `stabilized_tversky_index` **without**
   `alpha` or `beta`, so the symmetric defaults `alpha = beta = 0.5` apply and
   the term is a focal Dice loss with `gamma = 0.75`. The asymmetric
   0.3 / 0.7 weighting is not applied anywhere in this repository.
2. The boundary term filters the **thresholded** mask
   (`tf.cast(y_pred > 0.5, ...)`), so it carries no gradient with respect to
   the mask probabilities. It changes only the logged loss value (selection
   monitors `val_combined`), not boundary training. The curvature kernel uses `sigma = k/4 = 1.25` for
   the 5x5 filter.
3. The texture term applies Sobel to the **masks**, not to the ultrasound
   image, so it penalises horizontal mask-gradient dispersion rather than image
   texture, and the lesion-size factor `1 + clip(exp(-10 r), 0, 2)` attains its
   maximum at `r = 0`, that is on empty masks.

Each of these is documented at the relevant line in `loss.py`.

---

## Controls and audits

| Purpose | How |
|:--|:--|
| AIW controls | `config.AIW_MODE`: `adaptive` (reported), `off` (TIM output used directly), `fixed` (0.5 / 0.5), `constant` (two learned per-level constants, softmax, summing to 1) |
| Seeds | `python train.py --dataset busi --seed 1`; the split seed stays `config.RANDOM_STATE`; each run is saved as `checkpoints/best_<dataset>_<aiw_mode>_s<seed>.h5` and appended to `results/run_ledger.csv` (test N, correct count, early-stop flag, checkpoint hash) |
| BUSI-WHU | `python train.py --dataset busi_whu` after setting `WHU_IMAGE_GLOB` and `WHU_MASK_PATTERN` in `config.py` to the archive layout |
| Duplicate audit | `python audit_duplicates.py --root data/BUSI --manifest results/split_manifest_busi.csv` |
| Batch dependence | `python batch_check.py --checkpoint checkpoints/best_busi_adaptive_s42.h5 --sizes 1 8 24` |
| Cost | `python complexity.py` (parameters and FLOPs at 224 x 224) |

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_busi_adaptive_s42.h5
python evaluate.py --checkpoint checkpoints/best_busi_adaptive_s42.h5 --batch-size 1
```

`evaluate.py` is additive: it trains nothing and modifies nothing. It writes
three files to `config.RESULTS_DIR`:

| File | Contents |
|:-----|:---------|
| `predictions_<tag>.csv` | one row per test image: true and predicted class, all class probabilities, per-image TP/FP/FN, per-image IoU and Dice, and the four AIW weight pairs for that image |
| `confusion_<tag>.csv` | confusion matrix, entries sum to N |
| `summary_<tag>.json` | aggregates, each with its denominator: accuracy, weighted and macro F1, balanced accuracy, macro one-vs-rest AUC, per-class precision/recall/F1, mean and median Dice, mean IoU over positive-mask images, empty-mask false-positive rate |

It is built so the published aggregates can be checked rather than trusted:

* accuracy is recomputed as `100 x (integer correct) / N`, and the grid step
  `100/N` is printed, so any reported accuracy must be a multiple of it;
* IoU and Dice come from the same per-image counts at the same threshold, so
  the identity `Dice = 2 IoU / (1 + IoU)` holds per image and the aggregate
  obeys `mean Dice <= 2 mIoU / (1 + mIoU)`. The summary prints that ceiling and
  flags a violation;
* images with an empty ground-truth mask are counted separately instead of
  being silently dropped or scored as zero;
* AIW weights are recomputed as real model outputs through
  `AdaptiveInteractionWeighting.compute_weights`, so the exported per-level
  weights cover the whole test set rather than the last batch.

---

## Citation

Please cite the arXiv version (arXiv:2603.01295) until the camera-ready appears.

---

## License

MIT. See `LICENSE`.

## Acknowledgements

* **BUSI:** Al-Dhabyani et al., "Dataset of breast ultrasound images,"
  *Data in Brief* 28:104863, 2020.
* **EfficientNet:** Tan and Le, "EfficientNet: rethinking model scaling for
  convolutional neural networks," *ICML*, 2019.
* **Focal loss:** Lin et al., "Focal loss for dense object detection,"
  *ICCV*, 2017.
* **Attention U-Net:** Oktay et al., "Attention U-Net: learning where to look
  for the pancreas," arXiv:1804.03999, 2018.
