# Adaptive Bidirectional Task Interaction for Joint Segmentation and Classification of Breast Ultrasound

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation.

> **Note on naming.** This repository was previously released as
> *Uncertainty-Aware Multi-Level Decoder Interaction*. The adaptive weighting
> module computes activation-dispersion statistics, which are descriptive
> statistics of the activations rather than estimates of predictive
> uncertainty, so the module has been renamed and the uncertainty framing
> removed. The computation is unchanged.
>
> | Previous name | Current name |
> |---|---|
> | Uncertainty Proxy Attention (UPA) | Adaptive Interaction Weighting (AIW) |
> | Hierarchical Multi-Scale Fusion (HMSF) | Multi-Scale Context Fusion (MSCF) |
> | Task Interaction Module (TIM) | unchanged |

---

## Overview

Joint lesion segmentation and tissue classification are usually trained with a
shared encoder, so the two branches stop exchanging information once their
decoders separate. This work restores that exchange during decoding and makes
its strength an explicit, learned quantity.

- **TIM** exchanges information between the segmentation decoder and the
  classification branch at each of four decoder levels.
- **AIW** blends pre-interaction and post-interaction features with a
  coefficient computed per image and per level.
- **MSCF** applies three dilated separable convolutions with softmax scale
  competition inside every encoder stage and decoder block.

---

## Results

Trained and evaluated under the protocol in `config.py`. Segmentation is
foreground IoU and Dice at a threshold of 0.5; classification is accuracy and
weighted F1.

| Method | BUSI IoU | BUSI Dice | BUSI Acc | BUSI F1 | WHU IoU | WHU Dice | WHU Acc | WHU F1 |
|---|---|---|---|---|---|---|---|---|
| U-Net | 66.80 | 80.05 | 86.50 | 84.70 | 77.50 | 87.30 | 90.80 | 89.10 |
| Attention U-Net | 68.20 | 81.05 | 87.80 | 86.00 | 79.20 | 88.40 | 92.10 | 90.50 |
| UNet++ | 69.50 | 81.95 | 88.21 | 86.90 | 80.70 | 89.30 | 92.80 | 91.41 |
| TransUNet | 70.30 | 82.55 | – | – | 81.60 | 89.90 | – | – |
| Swin-UNet | 71.50 | 83.40 | – | – | 83.00 | 90.70 | – | – |
| MISSFormer | 72.80 | 84.25 | – | – | 84.60 | 91.60 | – | – |
| MTAN | 68.90 | 81.60 | 87.20 | 85.40 | 80.00 | 88.90 | 91.60 | 90.00 |
| MTANet | 72.10 | 83.80 | 89.30 | 87.60 | 83.70 | 91.10 | 93.90 | 92.30 |
| MTL-OCA | 72.90 | 84.30 | 89.90 | 88.20 | 84.50 | 91.60 | 94.50 | 92.90 |
| **Proposed** | **74.19** | **85.25** | **90.60** | **89.84** | **86.40** | **92.70** | **95.00** | **94.74** |

### Component ablation (BUSI)

| MSCF | TIM | AIW | IoU | Dice | Sens | Prec_s | Acc | F1 | AUC | Prec_c |
|:---:|:---:|:---:|---|---|---|---|---|---|---|---|
| – | – | – | 67.43 | 80.55 | 75.06 | 79.73 | 84.62 | 82.33 | 94.90 | 82.16 |
| ✓ | – | – | 69.12 | 81.74 | 79.12 | 80.19 | 88.89 | 88.34 | 96.30 | 86.70 |
| – | ✓ | – | 68.94 | 81.80 | 79.83 | 79.76 | 86.32 | 84.00 | 94.41 | 86.04 |
| – | ✓ | ✓ | 69.74 | 82.17 | **83.07** | 79.32 | 88.89 | 88.34 | 97.31 | 87.24 |
| **✓** | **✓** | **✓** | **74.19** | **85.25** | 80.87 | **85.47** | **90.60** | **89.84** | **97.66** | **89.95** |

- MSCF alone: +1.69 IoU. TIM alone: +1.51 IoU.
- Rows 3 and 4 differ only in AIW and isolate it: AUC 94.41 → 97.31,
  accuracy 86.32 → 88.89, IoU +0.80.
- The gains are not additive: MSCF and the TIM–AIW pair give 1.69 and 2.31
  separately, totalling 4.00, but 6.76 together.

---

## Installation

```bash
git clone https://github.com/C-loud-Nine/Adaptive-Task-Interaction-BUS.git
cd Adaptive-Task-Interaction-BUS
pip install -r requirements.txt
```

## Data

```
data/
├── BUSI/                 # 780 images, 3 classes
│   ├── normal/           # image.png + image_mask.png (several masks are merged)
│   ├── benign/
│   └── malignant/
└── BUSI-WHU/             # 927 images, 2 classes
    ├── benign/
    │   ├── images/
    │   └── masks/
    └── malignant/
```

BUSI: [Al-Dhabyani et al., Data in Brief 2020](https://doi.org/10.1016/j.dib.2019.104863)

BUSI-WHU: [Huang et al., Mendeley Data V3](https://doi.org/10.17632/k6cpmwybk3.3)

## Usage

```bash
# train on BUSI (set DATASET = 'busi_whu' in config.py for BUSI-WHU)
python train.py

# reproduce the reported metrics
python evaluate.py --weights checkpoints/best_model.h5 --dataset busi
```

`evaluate.py` is the script that produces the numbers in the tables above.
Note that the `MeanIoU` metric printed during training averages foreground and
background IoU on unthresholded outputs, so it is not comparable to the
foreground IoU reported in the paper; use `evaluate.py` for reporting.

---

## Repository layout

```
├── config.py         # paths, hyperparameters, dataset selection
├── data_loader.py    # BUSI and BUSI-WHU loading, splitting, augmentation
├── modules.py        # MSCF, DualPathAttention, ResidualBlock, AttentionGate, TIM, AIW
├── model.py          # encoder, 4-level decoder, dual heads
├── loss.py           # focal Tversky + boundary + texture; focal cross-entropy
├── train.py          # training loop and callbacks
└── evaluate.py       # reported metrics
```

---

## Notes on the implementation

These are stated so that the code and the paper can be read together.

- **The TIM modulation is one-sided.** It is `1 + 0.7 * sigmoid(.) * sigmoid(.)`,
  so the factor lies in `[1, 1.7]`: a channel is left unchanged or amplified,
  never attenuated. Reversion toward the original features is achieved by AIW
  driving its coefficient toward zero, not by the modulation itself.
- **AIW coefficients are normalised within the batch**, so a sample's
  coefficient depends on the other samples in its batch.
- **AIW uses a softmax**, so the two coefficients sum to one: the branches share
  a single interaction budget at each level.
- **BUSI partitions are image-level**, because the dataset carries no patient
  identifiers. BUSI is also known to contain
  [duplicated images](https://doi.org/10.1016/j.dib.2023.109162), which are not
  removed here.
- **The boundary loss term is computed on binarised masks**, as defined in the
  paper.

## Changes from the previous release

- Module renames (table at the top); computation unchanged.
- BUSI-WHU loading added. The previous release was BUSI-only, so the BUSI-WHU
  results could not be reproduced from it.
- `evaluate.py` added, computing the foreground IoU and Dice reported in the paper.
- The data split now reads `TRAIN/VAL/TEST_SPLIT` from `config.py`. The previous
  loader declared 70/15/15 in config but hard-coded 60/15/25, so the constants
  had no effect.
- The post-augmentation shuffle is now seeded.
- Removed an unused `task_context` branch in `DualPathAttention` that no call
  site invoked.
- README numbers corrected to match the paper.
- Architecture and qualitative figures temporarily removed pending relabelling;
  the previous versions carried the old module names.

## Citation

```bibtex
@article{shafi2026adaptive,
  title   = {Adaptive bidirectional task interaction for joint segmentation and
             classification of breast ultrasound},
  author  = {Al Shafi, Abdullah and Zunayed, Md Kawsar Mahmud Khan and
             Ahmmed, Safin and Hossain, Sk Imran and Mephu Nguifo, Engelbert},
  journal = {arXiv preprint arXiv:2603.01295},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
