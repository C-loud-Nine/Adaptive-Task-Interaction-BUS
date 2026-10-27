# Adaptive Bidirectional Task Interaction for Joint Segmentation and Classification of Breast Ultrasound

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation. A multi-task network exchanges latent features
between lesion segmentation and classification at every decoder level:

* **TIM (Task Interaction Module)**: a pooled decoder summary updates the
  classification vector through a gated residual, and the classification
  vector modulates the decoder channels.
* **AIW (Adaptive Interaction Weighting)**: a per-image, per-level blend
  between enhanced and original features, set from activation dispersion and
  trained through the task losses alone (0.84 M parameters for TIM and AIW).
* **MSCF (Multi-Scale Context Fusion)**: dilated separable convolutions
  (rates 1, 2, 4) fused by a per-image softmax over scales.

EfficientNet-B4 encoder (ImageNet), four-level decoder with attention-gated
skips; 57.3 M parameters, 18.4 GFLOPs per 224 x 224 image.

## Results

Mean and standard deviation over three seeds, test sets of the splits below.

| Dataset | IoU (%) | Dice (%) | Accuracy (%) | AUC (%) |
|:--|:--:|:--:|:--:|:--:|
| BUSI | 72.9 ± 0.4 | 82.5 ± 0.4 | 90.1 ± 0.5 | 92.6 ± 0.3 |
| BUSI-WHU | 85.4 ± 0.5 | 90.9 ± 0.4 | 91.1 ± 0.4 | 94.9 ± 0.3 |

IoU and Dice: image-wise means over lesion-positive test images at threshold
0.5. Accuracy and AUC (macro one-vs-rest on BUSI, binary on BUSI-WHU): all test
images.

## Installation

```bash
pip install -r requirements.txt
```

## Data

* **BUSI** (Al-Dhabyani et al., *Data in Brief* 28:104863, 2020): 780 images
  (133 normal, 437 benign, 210 malignant), <https://data.mendeley.com/datasets/wmy84gzngw/1>
* **BUSI-WHU v3** (Huang et al., Mendeley Data, doi:10.17632/k6cpmwybk3.3):
  927 benign and malignant images.

```bash
python prepare_data.py --input /path/to/downloads   # links both datasets into ./data and checks the counts
```

**Curation and split** (`data_loader.py`, run automatically):

* images and masks resized to 224 x 224; multi-lesion BUSI masks merged; images
  whose mask contradicts the class label are excluded;
* duplicate groups from the published BUSI duplicate list (Pawłowska et al.,
  *Data in Brief* 48:109247, 2023; `splits/busi_duplicate_groups.csv`) and from
  a near-duplicate check on both datasets (64 x 64 thumbnails, correlation
  ≥ 0.98); groups with conflicting class labels are dropped, every other group
  is kept whole within one partition. BUSI: 761 images (131/427/203); BUSI-WHU: 925;
* 70/15/15 split stratified by class with `StratifiedGroupKFold` (20 folds,
  `random_state = 42`), BUSI 533/114/114 and BUSI-WHU 647/139/139; the loader asserts that no group
  crosses partitions and writes the split (with its SHA-256) and the excluded
  files to the run folder;
* training images only, after the split: each image is kept plus three augmented copies
  (horizontal flip, rotation up to 15°).

## Training and testing

```bash
python train.py --dataset busi --seed 42
python train.py --dataset busi_whu --seed 42      # also tested on the BUSI lesion images (external test)
```

The external test uses the 616 curated BUSI lesion images that remain after
removing the 14 near-exact copies of BUSI-WHU images.

Each run writes to `runs/<dataset>_<variant>_s<seed>/`: the best checkpoint,
per-image test predictions (including the AIW coefficients), summaries, the
validation history, the split manifest and the logs. An interrupted run resumes
from its last finished epoch.

| Setting | Value |
|:--|:--|
| Optimiser | Adam (β1 0.91, β2 0.999), batch 24, gradient clipping at 1, mixed precision |
| Schedule | lr 3e-4, cosine decay with plateau reduction, up to 100 epochs |
| Encoder | ImageNet weights, frozen through the depthwise convolution of the fourth MBConv block |
| Loss | 0.82 L_seg (size-weighted focal Tversky + BCE + 0.15 edge) + 0.18 focal cross-entropy |
| Selection | from epoch 4: AIW calibrated on the training images, validation score 0.7 x pooled IoU + 0.3 x accuracy; best epoch restored; early stopping (patience 22, min delta 0.004) |

**Ablation and controls** (Table 2; each row with `--seed 42`, `43` and `44`):

```bash
python train.py --mscf off --tim off                 # 1  baseline (same-backbone row of Table 1)
python train.py --mscf on  --tim off                 # 2  + MSCF
python train.py --mscf off --tim on --aiw-mode off   # 3  + TIM
python train.py --mscf off --tim on                  # 4  + TIM + AIW
python train.py                                      # 5  full model
python train.py --aiw-mode off                       # 6  full, ω = 1 (no AIW)
python train.py --aiw-mode fixed                     # 7  full, ω = 0.5
python train.py --aiw-mode sigmoid                   # 8  full, independent sigmoid gates
```

## Comparison methods

`baselines/` contains the comparison methods, following their official
implementations, on the same split and with the same test metrics.

```bash
python baselines/train_baseline.py --model mti_net --dataset busi --seed 42 --out runs/busi_mti_net_s42
```

| Model | Backbone and initialisation |
|:--|:--|
| `unet`, `att_unet`, `unet_pp` | from scratch, with an added classification head |
| `transunet` | R50-ViT-B/16, ImageNet-21k; segmentation only |
| `swin_unet` | Swin-T, ImageNet-1k; segmentation only |
| `missformer` | official architecture, from scratch; segmentation only |
| `mtan` | official SegNet-MTAN, from scratch |
| `mtanet` | ResNet-50, ImageNet-1k |
| `mti_net` | HRNet-W18, ImageNet-1k |
| `mtl_oca` | Res-UNet with GroupNorm, from scratch |

## Statistics

```bash
python analysis/aggregate.py --runs runs --reference full --out stats \
    --others unet att_unet unet_pp transunet swin_unet missformer mtan mtanet mti_net mtl_oca
```

Collects all runs: means and standard deviations over seeds, and for the
proposed model against each comparison method the per-image IoU and accuracy
differences (averaged over seeds where there are several) with a paired
bootstrap that resamples whole duplicate groups (10,000 resamples), 95% CI and
p-values Holm-corrected over the ten comparisons, plus summary statistics of
the AIW coefficients.

## Repository structure

```
config.py               settings of the paper
prepare_data.py         links the downloaded datasets into ./data
data_loader.py          loading, curation, duplicate grouping, split, augmentation
model.py                TIM, AIW, MSCF, network, AIW calibration, losses, schedule
train.py                training, model selection, testing, complexity
metrics.py              test metrics and result files (all methods)
baselines/              comparison methods and their training script
analysis/aggregate.py   statistics over runs
splits/                 published BUSI duplicate groups
```

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

MIT, see [LICENSE](LICENSE).
