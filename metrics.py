"""
metrics.py - test metrics and result files, shared by train.py and
baselines/train_baseline.py, so every method is scored by the same code.

write_outputs() writes to results_dir:
  predictions_<tag>.csv   one row per test image: classes, probabilities,
                          per-image TP/FP/FN, IoU and Dice, and optional
                          extra per-image columns (e.g. AIW weights)
  confusion_<tag>.csv     confusion matrix (models with a classifier)
  summary_<tag>.json      aggregate metrics with their denominators

Segmentation: image-wise foreground IoU and Dice at threshold 0.5, averaged over test images with a non-empty ground-truth mask;
for images with an empty mask (BUSI normal) the share with any predicted lesion
pixel is reported. Classification: accuracy, macro-F1, balanced accuracy and
macro one-vs-rest AUC over all test images.
"""
import json
import os

import numpy as np

THRESHOLD = 0.5
EPS = 1e-7


def per_image_overlap(y_true, y_pred_prob, threshold=THRESHOLD):
    """Per-image TP, FP, FN, IoU and Dice over foreground pixels.

    Images with an empty ground-truth mask get NaN for IoU and Dice."""
    t = (y_true > 0.5).astype(np.float64).reshape(len(y_true), -1)
    p = (y_pred_prob > threshold).astype(np.float64).reshape(len(y_pred_prob), -1)
    tp = (t * p).sum(axis=1)
    fp = ((1.0 - t) * p).sum(axis=1)
    fn = (t * (1.0 - p)).sum(axis=1)
    has_fg = t.sum(axis=1) > 0
    iou = np.where(has_fg, tp / np.maximum(tp + fp + fn, EPS), np.nan)
    dice = np.where(has_fg, 2.0 * tp / np.maximum(2.0 * tp + fp + fn, EPS), np.nan)
    return tp, fp, fn, iou, dice, has_fg


def write_outputs(tag, dataset, checkpoint, batch_size, files, y_test, clf_prob,
                  seg_prob, mask_gt, cls, results_dir, extra=None, save_masks=False):
    """Score one model on the test set and write the result files.

    clf_prob is None for segmentation-only models. extra is an optional dict
    {column name: per-image array} appended to the per-image CSV (e.g. AIW
    weights). Returns the summary dict."""
    os.makedirs(results_dir, exist_ok=True)
    n = len(seg_prob)
    seg_prob = np.asarray(seg_prob)
    if seg_prob.ndim == 4:
        seg_prob = seg_prob[..., 0]
    mask_gt = np.asarray(mask_gt)
    if mask_gt.ndim == 4:
        mask_gt = mask_gt[..., 0]
    y_test = np.asarray(y_test).astype(int)
    assert len(mask_gt) == n == len(y_test) == len(files), "prediction length != N"
    extra = extra or {}
    for k, v in extra.items():
        assert len(v) == n, f"column {k} has {len(v)} rows, expected {n}"

    if save_masks:
        import cv2
        mdir = os.path.join(results_dir, f"masks_{tag}")
        os.makedirs(mdir, exist_ok=True)
        for i in range(n):
            name = os.path.splitext(str(files[i]).replace("/", "__").replace("\\", "__"))[0] + ".png"
            cv2.imwrite(os.path.join(mdir, name), ((seg_prob[i] > THRESHOLD) * 255).astype(np.uint8))

    tp, fp, fn, iou, dice, has_fg = per_image_overlap(mask_gt, seg_prob)

    has_clf = clf_prob is not None
    if has_clf:
        clf_prob = np.asarray(clf_prob)
        assert len(clf_prob) == n
        clf_pred = clf_prob.argmax(axis=1)
        correct = clf_pred == y_test

    # ---- per-image CSV ---------------------------------------------------
    import csv
    path = os.path.join(results_dir, f"predictions_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "file", "true_class", "pred_class", "correct", "max_prob"] +
                   [f"p_{c}" for c in cls] +
                   ["tp", "fp", "fn", "iou", "dice", "has_foreground"] + list(extra))
        for i in range(n):
            if has_clf:
                row = [i, files[i], cls[y_test[i]], cls[clf_pred[i]], int(correct[i]),
                       f"{clf_prob[i].max():.6f}"] + [f"{v:.6f}" for v in clf_prob[i]]
            else:
                row = [i, files[i], cls[y_test[i]], "", "", ""] + [""] * len(cls)
            row += [f"{tp[i]:.0f}", f"{fp[i]:.0f}", f"{fn[i]:.0f}",
                    f"{iou[i]:.6f}", f"{dice[i]:.6f}", int(has_fg[i])]
            row += [f"{float(extra[k][i]):.6f}" for k in extra]
            w.writerow(row)

    summary = {
        "checkpoint": checkpoint, "dataset": dataset, "eval_batch_size": batch_size,
        "threshold": THRESHOLD, "n_test_images": n,
        "n_positive_mask_images": int(has_fg.sum()),
        "n_empty_mask_images": int((~has_fg).sum()),
        "empty_mask_fp_rate_percent": (100.0 * float((fp[~has_fg] > 0).mean())
                                       if (~has_fg).any() else None),
        "empty_mask_fg_fraction_percent": (100.0 * float((fp[~has_fg] / mask_gt[0].size).mean())
                                           if (~has_fg).any() else None),
        "mean_iou_percent": float(np.nanmean(iou) * 100.0),
        "mean_dice_percent": float(np.nanmean(dice) * 100.0),
        "median_dice_percent": float(np.nanmedian(dice) * 100.0),
        "n_correct": None, "accuracy_percent": None, "macro_f1_percent": None,
        "balanced_accuracy_percent": None, "weighted_f1_percent": None,
        "macro_ovr_auc_percent": None, "per_class": None, "ece_percent": None,
    }

    if has_clf:
        k = len(cls)
        cm = np.zeros((k, k), dtype=int)
        for t_, p_ in zip(y_test, clf_pred):
            cm[t_, p_] += 1
        assert cm.sum() == n
        with open(os.path.join(results_dir, f"confusion_{tag}.csv"), "w") as f:
            f.write("true\\pred," + ",".join(cls) + "\n")
            for i, c in enumerate(cls):
                f.write(c + "," + ",".join(str(v) for v in cm[i]) + "\n")
        per_class, f1s, recs = {}, [], []
        for i, c in enumerate(cls):
            tp_c = int(cm[i, i]); fn_c = int(cm[i].sum()) - tp_c; fp_c = int(cm[:, i].sum()) - tp_c
            prec = tp_c / max(tp_c + fp_c, 1); rec = tp_c / max(tp_c + fn_c, 1)
            f1 = 2 * prec * rec / max(prec + rec, EPS)
            per_class[c] = {"support": int(cm[i].sum()), "precision": prec, "recall": rec, "f1": f1}
            f1s.append(f1); recs.append(rec)
        from sklearn.metrics import f1_score, roc_auc_score
        try:
            auc = (float(roc_auc_score(y_test, clf_prob[:, 1])) if k == 2 else
                   float(roc_auc_score(y_test, clf_prob, multi_class="ovr", average="macro")))
        except ValueError:
            auc = None
        conf = clf_prob.max(axis=1)
        bins = np.linspace(0, 1, 16)
        ece = sum(abs(correct[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
                  * ((conf > lo) & (conf <= hi)).mean()
                  for lo, hi in zip(bins[:-1], bins[1:]) if ((conf > lo) & (conf <= hi)).any())
        summary.update({
            "ece_percent": 100.0 * float(ece),
            "n_correct": int(correct.sum()),
            "accuracy_percent": 100.0 * float(correct.mean()),
            "macro_f1_percent": 100.0 * float(np.mean(f1s)),
            "balanced_accuracy_percent": 100.0 * float(np.mean(recs)),
            "weighted_f1_percent": 100.0 * float(f1_score(y_test, clf_pred, average="weighted")),
            "macro_ovr_auc_percent": (100.0 * auc if auc is not None else None),
            "per_class": per_class,
        })

    with open(os.path.join(results_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==================== TEST RESULTS ====================")
    if has_clf:
        print(f"  accuracy    = {summary['n_correct']}/{n} = {summary['accuracy_percent']:.4f}%")
        print(f"  macro-F1    = {summary['macro_f1_percent']:.4f}%")
        if summary["macro_ovr_auc_percent"] is not None:
            print(f"  macro AUC   = {summary['macro_ovr_auc_percent']:.4f}%")
    print(f"  mean IoU    = {summary['mean_iou_percent']:.4f}% over {summary['n_positive_mask_images']} lesion images")
    print(f"  mean Dice   = {summary['mean_dice_percent']:.4f}%")
    if summary["empty_mask_fp_rate_percent"] is not None:
        print(f"  normal images with any predicted lesion pixel = {summary['empty_mask_fp_rate_percent']:.2f}%")
    print("======================================================")
    return summary
