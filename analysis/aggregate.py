"""
aggregate.py - tables and paired significance tests from the per-run outputs.

Reads every results/summary_<tag>_bs24.json and predictions_<tag>_bs24.csv
under the given folders (one folder per run, as unzipped from the run
folders written by train.py and baselines/train_baseline.py --out), groups runs
by dataset and configuration, and writes

  aggregate_runs.csv        one row per run
  aggregate_table.csv       mean and sample standard deviation over seeds
  aggregate_tests.csv       reference configuration against every other one:
                            paired bootstrap over test images (10,000
                            resamples of whole duplicate groups) of the
                            difference in mean IoU and in accuracy, with 95% CI
                            and two-sided p-values, Holm-corrected over the
                            comparisons within each dataset
  aggregate_aiw.csv         AIW coefficients of the reference configuration:
                            per level, mean / std / min / max over test images,
                            and Spearman correlation with lesion area and with
                            per-image IoU (lesion-positive images)

The bootstrap resamples whole duplicate groups of test images; it describes
uncertainty due to the finite test set, with seed variation summarised by the
standard deviation. --others restricts the tests (and the Holm correction) to
the given configurations, e.g. the ten comparison methods of Table 1.

Per-image values are first averaged over the seeds of a configuration, then
compared image by image; all configurations must share the same test images
(checked through the file column).

External tests (BUSI-WHU models on BUSI lesion images, files
summary_ext-busi_<stem>_bs24.json) are grouped under dataset "busi_whu->busi".

Usage:
    python analysis/aggregate.py --runs path/to/unzipped_runs --reference full
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def run_key(stem):
    """'best_busi_full_s42' -> ('busi', 'full', 42)."""
    m = re.match(r"best_(busi_whu|busi)_(.+)_s(\d+)$", stem)
    if not m:
        raise ValueError(f"unrecognised checkpoint stem: {stem}")
    return m.group(1), m.group(2), int(m.group(3))


def load_runs(root):
    rows, preds = [], {}
    files = sorted(glob.glob(os.path.join(root, "**", "summary_*_bs*.json"), recursive=True))
    chosen = {}
    for sfile in files:                     # one evaluation per run: batch 24 if present
        m = re.match(r"summary_(.+)_bs(\d+)\.json$", os.path.basename(sfile))
        stem, bs = m.group(1), int(m.group(2))
        if stem not in chosen or bs == 24:
            chosen[stem] = (sfile, bs)
    for stem, (sfile, bs) in sorted(chosen.items()):
        ext = None
        if stem.startswith("ext-"):
            ext, stem_run = stem[4:].split("_", 1)
        else:
            stem_run = stem
        ds, cfg, seed = run_key(stem_run)
        if ext:
            ds = f"{ds}->{ext}"
        s = json.load(open(sfile))
        pfile = os.path.join(os.path.dirname(sfile), f"predictions_{stem}_bs{bs}.csv")
        p = pd.read_csv(pfile).set_index("file").sort_index()
        key = (ds, cfg, seed)
        if key in preds:
            raise ValueError(f"duplicate run {key}")
        preds[key] = p
        rows.append({"dataset": ds, "config": cfg, "seed": seed,
                     "iou": s["mean_iou_percent"], "dice": s["mean_dice_percent"],
                     "acc": s["accuracy_percent"], "auc": s["macro_ovr_auc_percent"],
                     "macro_f1": s["macro_f1_percent"], "ece": s.get("ece_percent"),
                     "bal_acc": s["balanced_accuracy_percent"],
                     "normal_fp_rate": s["empty_mask_fp_rate_percent"],
                     "normal_fp_area": s.get("empty_mask_fg_fraction_percent"),
                     "n_test": s["n_test_images"], "n_pos": s["n_positive_mask_images"]})
    return pd.DataFrame(rows), preds


def load_groups(root):
    """{dataset: {file: duplicate group}} from the split manifests under root."""
    out = {}
    for f in glob.glob(os.path.join(root, "**", "split_manifest_*.csv"), recursive=True):
        ds = re.match(r"split_manifest_(.+)\.csv$", os.path.basename(f)).group(1)
        if ds not in out:
            m = pd.read_csv(f)
            out[ds] = dict(zip(m["file"].astype(str), m["group"]))
    return out


def groups_for(groups, ds, files):
    """Group id per test file; the external set (busi_whu->busi) uses the BUSI groups."""
    g = groups.get(ds.split("->")[-1] if "->" in ds else ds)
    if g is None:
        return None
    return np.array([g.get(str(f), f"single:{f}") for f in files], dtype=object)


def per_image(preds, ds, cfg):
    """Seed-averaged per-image IoU (lesion-positive images) and correctness."""
    runs = [p for (d, c, _), p in preds.items() if d == ds and c == cfg]
    files = runs[0].index
    for p in runs[1:]:
        assert p.index.equals(files), f"{ds}/{cfg}: runs differ in test images"
    iou = np.nanmean(np.stack([p["iou"].to_numpy(float) for p in runs]), axis=0)
    corr = np.mean(np.stack([pd.to_numeric(p["correct"], errors="coerce").to_numpy(float)
                             for p in runs]), axis=0)
    return files, iou, corr


def bootstrap_diff(a, b, n_boot=10000, seed=0, groups=None):
    """Paired bootstrap of mean(a) - mean(b); returns diff, 95% CI, two-sided p.

    With groups (duplicate-group id per test image), whole groups are
    resampled, so near-copies of one lesion count as one unit (cluster
    bootstrap); the statistic is the image-level mean of the resampled groups."""
    rng = np.random.default_rng(seed)
    d = a - b
    ok = ~np.isnan(d)
    if groups is None:
        d = d[ok]
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots = d[idx].mean(axis=1)
    else:
        g = np.asarray(groups)[ok]
        d = d[ok]
        uniq, inv = np.unique(g, return_inverse=True)
        s = np.bincount(inv, weights=d)
        n = np.bincount(inv).astype(float)
        idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--reference", default="full")
    ap.add_argument("--others", nargs="*", default=None,
                    help="configurations to compare with the reference (default: all)")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs, preds = load_runs(args.runs)
    groups = load_groups(args.runs)
    if runs.empty:
        raise SystemExit(f"no summary_*.json files under {args.runs}")
    runs.to_csv(os.path.join(args.out, "aggregate_runs.csv"), index=False)
    metrics = ["iou", "dice", "acc", "auc", "macro_f1", "ece", "bal_acc",
               "normal_fp_rate", "normal_fp_area"]
    table = runs.groupby(["dataset", "config"])[metrics].agg(["mean", "std", "count"])
    table.to_csv(os.path.join(args.out, "aggregate_table.csv"))
    print(table.round(2).to_string())

    tests = []
    for ds in sorted(runs.dataset.unique()):
        cfgs = sorted(runs[runs.dataset == ds].config.unique())
        if args.reference not in cfgs:
            continue
        f_ref, iou_ref, cor_ref = per_image(preds, ds, args.reference)
        for cfg in cfgs:
            if cfg == args.reference or (args.others and cfg not in args.others):
                continue
            f, iou, cor = per_image(preds, ds, cfg)
            assert f.equals(f_ref), f"{ds}: {cfg} and {args.reference} differ in test images"
            gr = groups_for(groups, ds, f_ref)
            di, lo_i, hi_i, p_i = bootstrap_diff(iou_ref * 100, iou * 100, groups=gr)
            if np.isnan(cor).all() or np.isnan(cor_ref).all():      # segmentation-only model
                da = lo_a = hi_a = p_a = np.nan
            else:
                da, lo_a, hi_a, p_a = bootstrap_diff(cor_ref * 100, cor * 100, groups=gr)
            tests.append({"dataset": ds, "reference": args.reference, "other": cfg,
                          "d_iou": di, "d_iou_ci_low": lo_i, "d_iou_ci_high": hi_i,
                          "p_iou_bootstrap": p_i,
                          "d_acc": da, "d_acc_ci_low": lo_a, "d_acc_ci_high": hi_a,
                          "p_acc_bootstrap": p_a})
    tests = pd.DataFrame(tests)
    if len(tests):
        for col in ("p_iou_bootstrap", "p_acc_bootstrap"):
            tests[col + "_holm"] = np.nan
            for ds in tests.dataset.unique():
                m = (tests.dataset == ds) & tests[col].notna()
                tests.loc[m, col + "_holm"] = holm(tests.loc[m, col].to_numpy())
    tests.to_csv(os.path.join(args.out, "aggregate_tests.csv"), index=False)
    if len(tests):
        print(tests.round(4).to_string())

    # AIW coefficients of the reference configuration
    rows = []
    for (ds, cfg, seed), p in preds.items():
        if cfg != args.reference:
            continue
        pos = p["has_foreground"].astype(int) == 1
        area = (p["tp"] + p["fn"]).astype(float)
        for col in [c for c in p.columns if c.startswith("w_")]:
            w = p[col].astype(float)
            rows.append({"dataset": ds, "seed": seed, "coefficient": col,
                         "mean": w.mean(), "std": w.std(), "min": w.min(), "max": w.max(),
                         "spearman_area": spearmanr(w[pos], area[pos]).correlation if pos.sum() > 2 else np.nan,
                         "spearman_iou": spearmanr(w[pos], p.loc[pos, "iou"].astype(float)).correlation
                         if pos.sum() > 2 else np.nan})
    aiw = pd.DataFrame(rows)
    if len(aiw):
        aiw = aiw.groupby(["dataset", "coefficient"]).mean(numeric_only=True).drop(columns="seed")
        aiw.to_csv(os.path.join(args.out, "aggregate_aiw.csv"))
        print(aiw.round(4).to_string())


if __name__ == "__main__":
    main()
