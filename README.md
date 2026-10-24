# Adaptive Bidirectional Task Interaction for Joint Segmentation and Classification of Breast Ultrasound

<div align="center">

[![Python 3.9-3.11](https://img.shields.io/badge/python-3.9--3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official implementation**

</div>

---

## Overview

Lesion segmentation and classification in breast ultrasound are complementary:
lesion shape and margin, which segmentation delineates, are also cues to
malignancy. This repository implements a multi-task network that exchanges
latent features between the two tasks at every decoder level.

* **TIM (Task Interaction Module)**, applied at all four decoder levels:
  a pooled decoder summary updates the classification vector through a gated
  residual, and the classification vector modulates the decoder channels.
* **AIW (Adaptive Interaction Weighting)**, a per-image, per-level scalar
  blend between enhanced and original features, set from activation
  dispersion and trained through the task losses alone.
* **MSCF (Multi-Scale Context Fusion)**, dilated separable convolutions
  (rates 1, 2, 4) fused by a per-image softmax over scales, in the encoder and
  decoder residual blocks.

## Results

Proposed model, mean of three training seeds on the released partition
(see the paper for all baselines and the ablation).

| Dataset | IoU (%) | D_J (%) | Accuracy (%) |
|:--|:--:|:--:|:--:|
| BUSI | 74.47 | 85.37 | 90.60 |
| BUSI-WHU | 85.97 | 92.46 | 94.52 |

On a BUSI partition that keeps each near-duplicate image group (Pawłowska et al.,
Data in Brief 48:109247, 2023) within one split, the proposed model reaches
73.58% IoU and 88.32% accuracy (mean of three seeds); the file lists of this
partition are in `splits/`.

IoU is the mean image-wise foreground IoU over lesion-positive test images at
threshold 0.5; D_J = 2J/(1+J) for the mean IoU J; accuracy is computed over all
test images.

---

## Architecture

EfficientNet-B4 encoder (ImageNet, first 50 layers frozen) and a four-level
decoder with transposed convolutions and attention-gated skip connections.
A TIM-AIW unit follows each decoder level.

| Decoder level | Resolution | Channels | Classification vector from |
|:-------------:|:----------:|:--------:|:---------------------------|
| D1 | 14x14 | 384 | bridge |
| D2 | 28x28 | 192 | encoder stage 4 |
| D3 | 56x56 | 96 | encoder stage 3 |
| D4 | 112x112 | 48 | encoder stage 2 |

The classification head concatenates pooled features of four encoder stages
with the four blended classification vectors and reduces them to 256 units
before the softmax.

### Task Interaction Module

```
Input:  decoder features D_l (B x H x W x C), classification vector f_l (B x 256)

# segmentation -> classification
s_l     = BN(ReLU(W_s GAP(ReLU(Conv1x1(D_l)))))
f_l^enh = f_l + sigmoid(W_g s_l) * s_l

# classification -> segmentation
p_l     = sigmoid(W_p f_l)
g_l     = sigmoid(W_q GAP(D_l))
D_l^enh = D_l * (1 + 0.7 * g_l * p_l)
```

### Adaptive Interaction Weighting

```
v_seg  = mean_c Var_hw(D_l^enh)          # activation dispersion
v_clf  = Var(f_l^enh)
v~_t   = v_t / (m_t + 1e-8)               # t in {seg, clf}
# m_t: batch mean of v_t during training; after training, train.py fixes m_t
#      to the mean of v_t over the training set computed in inference mode,
#      level by level, so test predictions do not depend on the batch
[w_seg, w_clf] = softmax(W_2 ReLU(W_1 [v~_seg, v~_clf]))
D_l^fin = D_l + w_seg * (D_l^enh - D_l)
f_l^fin = f_l + w_clf * (f_l^enh - f_l)
```

### Multi-Scale Context Fusion

```
F_r = SepConv3x3_dilation=r(X),  r in {1, 2, 4}
a   = softmax(W_3 ReLU(W_4 GAP(X)))
Y   = Conv1x1(sum_r a_r * F_r)
```

---

## Repository structure

```
.
|-- config.py        # hyperparameters, dataset paths, split ratios
|-- data_loader.py   # BUSI and BUSI-WHU loading, augmentation, splitting
|-- modules.py       # TIM, AIW, MSCF, attention gate, residual block
|-- model.py         # full network
|-- loss.py          # segmentation and classification losses
|-- train.py         # training (warm-up, cosine schedule, early stopping)
|-- evaluate.py      # test-set evaluation and per-image export
|-- complexity.py    # parameters and FLOPs
|-- requirements.txt
`-- README.md
```

---

## Data

**BUSI** (Al-Dhabyani et al., *Data in Brief* 28:104863, 2020):
780 images, 133 normal, 437 benign, 210 malignant.
<https://data.mendeley.com/datasets/wmy84gzngw/1>

**BUSI-WHU v3** (Huang et al., Mendeley Data, doi:10.17632/k6cpmwybk3.3):
927 benign and malignant images, labelled by the class folders of the release.
Set `WHU_IMAGE_GLOB` and `WHU_MASK_PATTERN` in `config.py` to the layout of the
downloaded archive.

```
data/
|-- BUSI/       normal/  benign/  malignant/
`-- BUSI-WHU/   benign/  malignant/
```

**Preprocessing and split**

* images and masks resized to 224x224 (masks by nearest neighbour, binarised);
  multi-lesion BUSI masks merged into one; normal images carry an empty mask;
* images scaled to [0, 1] (`config.INPUT_SCALE = "unit"`);
* stratified image-level 70/15/15 split with `random_state = 42`:
  546/117/117 (BUSI) and 648/139/140 (BUSI-WHU); the partition is written to
  `results/split_manifest_<dataset>.csv`;
* training images only: three augmented copies per image plus the original,
  with horizontal and vertical flips (p = 0.5 each) and rotations up to
  15 degrees (p = 0.7).

---

## Training

The code also runs on TensorFlow >= 2.16 with `tf-keras` and `TF_USE_LEGACY_KERAS=1`
(as in the Kaggle notebooks).

```bash
pip install -r requirements.txt
python train.py --dataset busi --seed 42
python train.py --dataset busi_whu --seed 42
```

| Setting | Value |
|:--|:--|
| Input | 224 x 224 x 3 |
| Encoder | EfficientNet-B4, ImageNet weights, first 50 layers frozen |
| Decoder channels | 384 / 192 / 96 / 48, dropout 0.3 |
| Batch size | 24 (training and evaluation) |
| Optimiser | Adam (beta1 = 0.91, beta2 = 0.999), global clipnorm 1.0 |
| Learning rate | 3e-4, cosine decay with plateau reduction (factor 0.55, patience 7) down to 1.5e-6 |
| Epochs | up to 100, including a 3-epoch warm-up |
| Early stopping | composite validation score 0.7 x two-class Keras MeanIoU of the sigmoid outputs + 0.3 x accuracy, patience 22, min delta 0.004 |
| Loss | 0.82 L_seg + 0.18 L_clf |

**Loss.** L_seg = mean_i[(1 + exp(-10 rho_i)) L_FT,i] + 0.15 L_edge, where
L_FT,i is the per-image focal Tversky loss (exponent 0.75, alpha = beta = 0.5),
rho_i the foreground fraction of image i, and L_edge an edge-statistics term on
the target and predicted masks. L_clf is focal cross-entropy (gamma = 2).

Each run is saved as `checkpoints/best_<dataset>_<variant>_s<seed>.keras`
(`full` for the paper's model) and logged in `results/run_ledger.csv`.

**Ablation (Table 2)**

```bash
python train.py --dataset busi --mscf off --tim off              # row 1
python train.py --dataset busi --mscf on  --tim off              # row 2
python train.py --dataset busi --mscf off --tim on --aiw-mode off   # row 3
python train.py --dataset busi --mscf off --tim on               # row 4
python train.py --dataset busi                                   # row 5 (full)
```

Without MSCF the residual blocks keep their two separable convolutions and
skip the MSCF layer; without TIM the decoder features and classification
vectors pass on unchanged; `--aiw-mode off` uses the TIM outputs directly.

---

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_busi_full_s42.keras
python evaluate.py --checkpoint checkpoints/best_busi_whu_full_s42.keras --dataset busi_whu
python complexity.py      # 57.4 M parameters, 18.5 GFLOPs per 224x224 image
```

`evaluate.py` writes per-image predictions (including the AIW weights of each
level), the confusion matrix and a JSON summary to `results/`.

---

## Citation

```bibtex
@misc{shafi2026adaptive,
  title  = {Adaptive Bidirectional Task Interaction for Joint Segmentation and
            Classification of Breast Ultrasound},
  author = {Shafi, Abdullah Al and Zunayed, Md Kawsar Mahmud Khan and
            Ahmmed, Safin and Hossain, Sk. Imran and Mephu Nguifo, Engelbert},
  year   = {2026},
  eprint = {2603.01295},
  archivePrefix = {arXiv}
}
```

## License

MIT. See `LICENSE`.

## Acknowledgements

BUSI (Al-Dhabyani et al., 2020), BUSI-WHU (Huang et al., 2025),
EfficientNet (Tan and Le, ICML 2019), Attention U-Net (Oktay et al., 2018),
focal loss (Lin et al., ICCV 2017), focal Tversky loss (Abraham and Khan,
ISBI 2019).
