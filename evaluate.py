"""
Evaluation: computes the metrics reported in the paper.

Segmentation : foreground IoU, Dice, sensitivity, precision (threshold 0.5)
Classification: accuracy, weighted F1, AUC, per-class report

Images with an empty ground-truth mask (BUSI 'normal') are excluded from the
segmentation metrics and counted separately, since foreground overlap is
undefined for them. They are retained for classification.

Usage
-----
    python evaluate.py --weights checkpoints/best_model.h5
    python evaluate.py --weights checkpoints/whu.h5 --dataset busi_whu
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)

import config
import data_loader
from model import adaptive_task_interaction_model


def foreground_iou_dice(y_true, y_pred, threshold=0.5):
    """Foreground IoU and Dice for one image. Returns (nan, nan) if the mask is empty."""
    t = (np.asarray(y_true).squeeze() > 0.5)
    p = (np.asarray(y_pred).squeeze() > threshold)
    if t.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    inter = np.logical_and(t, p).sum()
    union = np.logical_or(t, p).sum()
    iou = inter / union if union > 0 else np.nan
    denom = t.sum() + p.sum()
    dice = 2.0 * inter / denom if denom > 0 else np.nan
    sens = inter / t.sum()
    prec = inter / p.sum() if p.sum() > 0 else np.nan
    return iou, dice, sens, prec


def evaluate_segmentation(masks_true, masks_pred, threshold=0.5):
    ious, dices, sens, precs = [], [], [], []
    n_empty = 0
    for t, p in zip(masks_true, masks_pred):
        if (np.asarray(t).squeeze() > 0.5).sum() == 0:
            n_empty += 1
            continue
        i, d, s, pr = foreground_iou_dice(t, p, threshold)
        ious.append(i); dices.append(d); sens.append(s); precs.append(pr)
    return {
        'iou': float(np.nanmean(ious) * 100),
        'dice': float(np.nanmean(dices) * 100),
        'sensitivity': float(np.nanmean(sens) * 100),
        'precision': float(np.nanmean(precs) * 100),
        'n_evaluated': len(ious),
        'n_empty_masks_excluded': n_empty,
        'threshold': threshold,
    }


def evaluate_classification(y_true, y_prob, class_names):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    n_classes = np.asarray(y_prob).shape[1]
    try:
        auc = (roc_auc_score(y_true, y_prob[:, 1]) if n_classes == 2
               else roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except ValueError:
        auc = float('nan')
    return {
        'accuracy': float(accuracy_score(y_true, y_pred) * 100),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted') * 100),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro') * 100),
        'auc': float(auc * 100),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'per_class': classification_report(y_true, y_pred,
                                           target_names=class_names,
                                           output_dict=True, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='path to trained weights')
    ap.add_argument('--dataset', default=config.DATASET, choices=['busi', 'busi_whu'])
    ap.add_argument('--threshold', type=float, default=config.SEG_THRESHOLD)
    ap.add_argument('--out', default=os.path.join(config.RESULTS_DIR, 'evaluation.json'))
    args = ap.parse_args()

    categories = (config.BUSI_CATEGORIES if args.dataset == 'busi'
                  else config.WHU_CATEGORIES)

    data = data_loader.prepare_datasets(dataset=args.dataset)
    model = adaptive_task_interaction_model(
        input_size=config.INPUT_SIZE,
        num_seg_classes=config.NUM_SEG_CLASSES,
        num_clf_classes=len(categories),
        dropout_rate=config.DROPOUT_RATE,
    )
    model.load_weights(args.weights)

    seg_pred, clf_pred = model.predict(data['X_test'], batch_size=config.VAL_BATCH_SIZE)

    results = {
        'dataset': args.dataset,
        'n_test': int(len(data['X_test'])),
        'segmentation': evaluate_segmentation(data['mask_test'], seg_pred, args.threshold),
        'classification': evaluate_classification(data['y_test'], clf_pred, categories),
    }

    print(json.dumps({k: v for k, v in results.items() if k != 'classification'}, indent=2))
    print("accuracy      : %.2f" % results['classification']['accuracy'])
    print("weighted F1   : %.2f" % results['classification']['weighted_f1'])
    print("AUC           : %.2f" % results['classification']['auc'])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == '__main__':
    main()
