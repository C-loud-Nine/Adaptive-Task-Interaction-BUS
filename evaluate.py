"""
evaluate.py - per-image evaluation export for the breast ultrasound model.

ADDITIVE. Trains nothing and changes nothing in the model or the data
pipeline. Run it against an existing checkpoint to answer the question that
blocks submission: do the reported table values reconstruct from integer
counts and per-image metrics?

What it writes to config.RESULTS_DIR:
  predictions_<tag>.csv   one row per test image, including the four AIW
                          weight pairs for that image
  confusion_<tag>.csv     confusion matrix, entries sum to N
  summary_<tag>.json      aggregates, each stated with its denominator

Why each column exists:
  * Accuracy must equal 100 * (integer correct) / N, so every reported
    accuracy has to be a multiple of 100/N. The CSV makes the integer count
    and the denominator visible instead of inferred.
  * IoU and Dice are computed from the SAME per-image TP/FP/FN at the SAME
    threshold, so the per-image identity Dice = 2*IoU/(1+IoU) holds exactly
    and the aggregate obeys mean Dice <= 2*mIoU/(1+mIoU). The summary prints
    that ceiling and flags a violation.
  * Empty-ground-truth cases are reported separately rather than silently
    dropped or counted as zero, because that choice alone moves the mean.
  * AIW normalises its two dispersion statistics over the CURRENT BATCH, so
    the evaluation batch size changes the predictions. It is recorded in the
    summary and in the default tag.

Usage:
    python evaluate.py --checkpoint checkpoints/best_busi_adaptive_s42.h5
    python evaluate.py --checkpoint checkpoints/best_busi_adaptive_s42.h5 --batch-size 1
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

import config
import data_loader
import modules

THRESHOLD = 0.5
EPS = 1e-7

CUSTOM_OBJECTS = {
    "MultiScaleContextFusion": modules.MultiScaleContextFusion,
    "DualPathAttention": modules.DualPathAttention,
    "ResidualBlock": modules.ResidualBlock,
    "AttentionGate": modules.AttentionGate,
    "TaskInteractionModule": modules.TaskInteractionModule,
    "AdaptiveInteractionWeighting": modules.AdaptiveInteractionWeighting,
}


def per_image_overlap(y_true, y_pred_prob, threshold=THRESHOLD):
    """Return per-image TP, FP, FN, IoU, Dice over FOREGROUND pixels.

    IoU and Dice come from one set of counts, so they cannot disagree. An
    empty ground truth is reported as NaN rather than 1.0 or 0.0: it has no
    foreground to overlap, and forcing a value is what makes aggregate IoU and
    Dice drift apart.
    """
    t = (y_true > 0.5).astype(np.float64).reshape(len(y_true), -1)
    p = (y_pred_prob > threshold).astype(np.float64).reshape(len(y_pred_prob), -1)

    tp = (t * p).sum(axis=1)
    fp = ((1.0 - t) * p).sum(axis=1)
    fn = (t * (1.0 - p)).sum(axis=1)

    has_fg = t.sum(axis=1) > 0
    iou = np.where(has_fg, tp / np.maximum(tp + fp + fn, EPS), np.nan)
    dice = np.where(has_fg, 2.0 * tp / np.maximum(2.0 * tp + fp + fn, EPS), np.nan)
    return tp, fp, fn, iou, dice, has_fg


def _aiw_call_args(layer):
    """Return the four tensors an AIW layer was called with, in call order.

    Keras stores a multi-input call as node.call_args (positional) and
    node.call_kwargs (keyword); after loading a saved model the trailing
    arguments may appear as keywords. layer.input only returns the first
    positional argument, so it cannot be used here.
    """
    node = layer._inbound_nodes[0]
    args = list(node.call_args)
    kwargs = node.call_kwargs or {}
    for key in ("seg_base", "seg_enhanced", "clf_base", "clf_enhanced"):
        if key in kwargs:
            args.append(kwargs[key])
    return args


def build_weight_export_model(model):
    """Expose the AIW weights of every level as real model outputs.

    Uses AdaptiveInteractionWeighting.compute_weights on the tensors the layer
    was called with, so the exported values are the ones the blend actually
    used. Returns (export_model, level_names). level_names is empty if the AIW
    layers cannot be located, in which case the weight columns are omitted
    rather than invented.
    """
    names, outs = [], []
    for lvl in ["d1", "d2", "d3", "d4"]:
        try:
            layer = model.get_layer(f"aiw_{lvl}")
        except ValueError:
            print(f"[warn] layer 'aiw_{lvl}' not found; AIW weights not exported")
            return model, []
        ins = _aiw_call_args(layer)
        if len(ins) != 4:
            print(f"[warn] unexpected inputs on 'aiw_{lvl}'; AIW weights not exported")
            return model, []
        outs.append(layer.compute_weights(ins[1], ins[3]))   # (seg_enh, clf_enh)
        names.append(lvl)
    return tf.keras.Model(model.inputs, list(model.outputs) + outs), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dataset", default="busi", choices=["busi", "busi_whu"])
    ap.add_argument("--batch-size", type=int, default=config.VAL_BATCH_SIZE,
                    help="AIW normalises over the evaluation batch, so this "
                         "value changes the predictions. It is recorded.")
    args = ap.parse_args()
    # default tag follows the checkpoint, so runs and seeds do not overwrite each other
    tag = args.tag or f"{os.path.splitext(os.path.basename(args.checkpoint))[0]}_bs{args.batch_size}"

    print(f"[load] {args.checkpoint}")
    model = tf.keras.models.load_model(
        args.checkpoint, compile=False, custom_objects=CUSTOM_OBJECTS)

    path = config.BUSI_WHU_PATH if args.dataset == "busi_whu" else config.BUSI_PATH
    data = data_loader.prepare_datasets(path, dataset=args.dataset)
    X_test = data["X_test"]
    y_test = np.asarray(data["y_test"]).astype(int)
    mask_gt = np.asarray(data["mask_test"])
    if mask_gt.ndim == 4:
        mask_gt = mask_gt[..., 0]
    files = data.get("paths_test", np.array([str(i) for i in range(len(X_test))]))
    n = len(X_test)
    print(f"[data] test images N = {n}  (grid step = {100.0 / n:.4f} accuracy points)")

    export, levels = build_weight_export_model(model)
    preds = export.predict(X_test, batch_size=args.batch_size, verbose=1)

    seg_prob = np.asarray(preds[0])
    if seg_prob.ndim == 4:
        seg_prob = seg_prob[..., 0]
    clf_prob = np.asarray(preds[1])
    weights = [np.asarray(w).reshape(len(X_test), -1) for w in preds[2:]] if levels else []

    # Hard guarantee: nothing silently truncated to the last batch.
    assert len(seg_prob) == n and len(clf_prob) == n, "prediction length != N"
    for lvl, w in zip(levels, weights):
        assert len(w) == n, f"aiw_{lvl} has {len(w)} rows, expected {n}"

    clf_pred = clf_prob.argmax(axis=1)
    tp, fp, fn, iou, dice, has_fg = per_image_overlap(mask_gt, seg_prob)
    correct = (clf_pred == y_test)
    n_correct = int(correct.sum())

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    cls = config.WHU_CLASSES if args.dataset == "busi_whu" else config.CATEGORIES

    # ---- per-image CSV ---------------------------------------------------
    path = os.path.join(config.RESULTS_DIR, f"predictions_{tag}.csv")
    wcols = [f"w_{k}_{lvl}" for lvl in levels for k in ("seg", "clf")]
    with open(path, "w") as f:
        f.write("index,file,true_class,pred_class,correct,max_prob," +
                ",".join(f"p_{c}" for c in cls) +
                ",tp,fp,fn,iou,dice,has_foreground" +
                ("," + ",".join(wcols) if wcols else "") + "\n")
        for i in range(n):
            row = [str(i), str(files[i]), cls[y_test[i]], cls[clf_pred[i]], str(int(correct[i])),
                   f"{clf_prob[i].max():.6f}"]
            row += [f"{v:.6f}" for v in clf_prob[i]]
            row += [f"{tp[i]:.0f}", f"{fp[i]:.0f}", f"{fn[i]:.0f}",
                    f"{iou[i]:.6f}", f"{dice[i]:.6f}", str(int(has_fg[i]))]
            for w in weights:
                row += [f"{v:.6f}" for v in w[i]]
            f.write(",".join(row) + "\n")
    print(f"[write] {path}")

    # ---- confusion matrix ------------------------------------------------
    k = len(cls)
    cm = np.zeros((k, k), dtype=int)
    for t_, p_ in zip(y_test, clf_pred):
        cm[t_, p_] += 1
    assert cm.sum() == n, "confusion matrix does not sum to N"
    cpath = os.path.join(config.RESULTS_DIR, f"confusion_{tag}.csv")
    with open(cpath, "w") as f:
        f.write("true\\pred," + ",".join(cls) + "\n")
        for i, c in enumerate(cls):
            f.write(c + "," + ",".join(str(v) for v in cm[i]) + "\n")
    print(f"[write] {cpath}")

    # ---- aggregates, each with its denominator ---------------------------
    per_class, f1s, recs = {}, [], []
    for i, c in enumerate(cls):
        tp_c = int(cm[i, i])
        fn_c = int(cm[i].sum()) - tp_c
        fp_c = int(cm[:, i].sum()) - tp_c
        prec = tp_c / max(tp_c + fp_c, 1)
        rec = tp_c / max(tp_c + fn_c, 1)
        f1 = 2 * prec * rec / max(prec + rec, EPS)
        per_class[c] = {"support": int(cm[i].sum()), "precision": prec,
                        "recall": rec, "f1": f1}
        f1s.append(f1)
        recs.append(rec)

    from sklearn.metrics import f1_score, roc_auc_score
    weighted_f1 = float(f1_score(y_test, clf_pred, average="weighted"))
    try:
        auc = (float(roc_auc_score(y_test, clf_prob[:, 1])) if k == 2 else
               float(roc_auc_score(y_test, clf_prob, multi_class="ovr", average="macro")))
    except ValueError:   # a class absent from the test labels
        auc = None

    n_fg = int(has_fg.sum())
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "eval_batch_size": args.batch_size,
        "threshold": THRESHOLD,
        "n_test_images": n,
        "n_correct": n_correct,
        "accuracy_percent": 100.0 * n_correct / n,
        "accuracy_grid_step_percent": 100.0 / n,
        "macro_f1_percent": 100.0 * float(np.mean(f1s)),
        "balanced_accuracy_percent": 100.0 * float(np.mean(recs)),
        "weighted_f1_percent": 100.0 * weighted_f1,
        "macro_ovr_auc_percent": (100.0 * auc if auc is not None else None),
        "n_positive_mask_images": n_fg,
        "n_empty_mask_images": int((~has_fg).sum()),
        "empty_mask_fp_rate_percent": (100.0 * float((fp[~has_fg] > 0).mean())
                                       if (~has_fg).any() else None),
        "empty_mask_fg_fraction_percent": (100.0 * float((fp[~has_fg] / mask_gt[0].size).mean())
                                           if (~has_fg).any() else None),
        "mean_iou_percent": float(np.nanmean(iou) * 100.0),
        "mean_dice_percent": float(np.nanmean(dice) * 100.0),
        "median_dice_percent": float(np.nanmedian(dice) * 100.0),
        "per_class": per_class,
        "aiw_levels_exported": levels,
    }
    m = summary["mean_iou_percent"] / 100.0
    summary["max_mean_dice_implied_by_mean_iou_percent"] = 100.0 * 2 * m / (1 + m)
    summary["dice_within_ceiling"] = bool(
        summary["mean_dice_percent"]
        <= summary["max_mean_dice_implied_by_mean_iou_percent"] + 1e-9)
    if levels:
        summary["mean_aiw_weights"] = {
            f"{lvl}": {"w_seg": float(w[:, 0].mean()), "w_clf": float(w[:, 1].mean())}
            for lvl, w in zip(levels, weights)}

    spath = os.path.join(config.RESULTS_DIR, f"summary_{tag}.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ RECONSTRUCTION CHECK ================")
    print(f"  accuracy    = {n_correct}/{n} = {summary['accuracy_percent']:.4f}%")
    print(f"  grid step   = {summary['accuracy_grid_step_percent']:.4f} points")
    print(f"                any reported accuracy must be a multiple of this")
    print(f"  macro-F1    = {summary['macro_f1_percent']:.4f}%")
    print(f"  balanced acc= {summary['balanced_accuracy_percent']:.4f}%")
    print(f"  weighted F1 = {summary['weighted_f1_percent']:.4f}%")
    if auc is not None:
        print(f"  macro AUC   = {summary['macro_ovr_auc_percent']:.4f}% (one-vs-rest)")
    print(f"  mean IoU    = {summary['mean_iou_percent']:.4f}% over {n_fg} positive-mask images")
    print(f"  mean Dice   = {summary['mean_dice_percent']:.4f}%  "
          f"(ceiling {summary['max_mean_dice_implied_by_mean_iou_percent']:.4f}%, "
          f"{'OK' if summary['dice_within_ceiling'] else 'VIOLATED'})")
    print(f"  AIW levels exported: {levels or 'none'}")
    print("======================================================")
    print(f"[write] {spath}")


if __name__ == "__main__":
    main()
