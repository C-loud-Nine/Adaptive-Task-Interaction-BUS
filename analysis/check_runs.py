"""
check_runs.py - check every finished run against the fixed partitions and print the
values that go into the paper's tables.

    python analysis/check_runs.py --runs runs

--runs is a folder holding the run folders (or the unzipped *_results.zip files).
For every run it checks:
  * the run used the fixed partition: its results/split_manifest_<dataset>.csv has
    the same sha256 as splits/split_manifest_<dataset>.csv;
  * the test set: BUSI 115 images (94 lesion, 21 normal), BUSI-WHU 143 (all lesion),
    external test 630 BUSI lesion images, and exactly the test files of the split file;
  * the summary agrees with the per-image file: accuracy = correct / N, mean IoU and
    mean Dice over the lesion images, the normal-image count.
Then, per configuration, it prints mean and sample std over seeds rounded as in the
tables, the accuracy as k / (seeds x N), and lists runs that are missing.
A line starting with FAIL needs attention; the last line says whether all checks passed.
"""
import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "splits")
EXPECT = {"busi": (115, 94, 21), "busi_whu": (143, 143, 0), "busi_whu->busi": (630, 630, 0)}
ABLATION = {   # Table 2 row -> configuration name written by train.py
    1: "mscf-off_tim-off_aiw-adaptive", 2: "mscf-on_tim-off_aiw-adaptive",
    3: "mscf-off_tim-on_aiw-off", 4: "mscf-off_tim-on_aiw-adaptive", 5: "full",
    6: "mscf-on_tim-on_aiw-off", 7: "mscf-on_tim-on_aiw-fixed", 8: "mscf-on_tim-on_aiw-sigmoid"}
COMPARISON = ["unet", "att_unet", "unet_pp", "transunet", "swin_unet", "missformer",
              "mtan", "mtanet", "mti_net", "mtl_oca"]
SEG_ONLY = {"transunet", "swin_unet", "missformer"}
FAILS = []


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fail(msg):
    FAILS.append(msg)
    print("FAIL", msg)


def run_key(stem):
    m = re.match(r"best_(busi_whu|busi)_(.+)_s(\d+)$", stem)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    frozen = {ds: sha(os.path.join(SPLITS, f"split_manifest_{ds}.csv")) for ds in ("busi", "busi_whu")}
    test_files = {ds: set(pd.read_csv(os.path.join(SPLITS, f"split_manifest_{ds}.csv"))
                          .query("split == 'test'")["file"].astype(str)) for ds in ("busi", "busi_whu")}
    print("fixed partitions:", {k: v[:12] for k, v in frozen.items()})

    chosen = {}
    for sfile in sorted(glob.glob(os.path.join(args.runs, "**", "summary_*_bs*.json"), recursive=True)):
        m = re.match(r"summary_(.+)_bs(\d+)\.json$", os.path.basename(sfile))
        stem, bs = m.group(1), int(m.group(2))
        if stem not in chosen or bs == 24:
            chosen[stem] = (sfile, bs)

    rows = []
    for stem, (sfile, bs) in sorted(chosen.items()):
        ext = stem.startswith("ext-")
        key = run_key(stem[4:].split("_", 1)[1] if ext else stem)
        if key is None:
            fail(f"{stem}: unrecognised run name")
            continue
        ds_train, cfg, seed = key
        ds = f"{ds_train}->busi" if ext else ds_train
        name = f"{ds}/{cfg}/s{seed}"
        res = os.path.dirname(sfile)
        s = json.load(open(sfile))
        p = pd.read_csv(os.path.join(res, f"predictions_{stem}_bs{bs}.csv"))

        man = os.path.join(res, f"split_manifest_{ds_train}.csv")
        if not os.path.exists(man):
            fail(f"{name}: no split_manifest_{ds_train}.csv next to the results")
        elif sha(man) != frozen[ds_train]:
            fail(f"{name}: trained on a different partition (sha256 {sha(man)[:12]}, expected {frozen[ds_train][:12]})")

        n, n_pos, n_neg = EXPECT[ds]
        files = set(p["file"].astype(str))
        if len(p) != n or s["n_test_images"] != n:
            fail(f"{name}: {len(p)} test images, expected {n}")
        if not ext and files != test_files[ds]:
            fail(f"{name}: test files differ from the split file ({len(files ^ test_files[ds])} differ)")
        pos = p["has_foreground"].astype(int) == 1
        if int(pos.sum()) != n_pos or int((~pos).sum()) != n_neg:
            fail(f"{name}: {int(pos.sum())} lesion / {int((~pos).sum())} normal, expected {n_pos} / {n_neg}")

        iou, dice = 100 * p.loc[pos, "iou"].astype(float).mean(), 100 * p.loc[pos, "dice"].astype(float).mean()
        if abs(iou - s["mean_iou_percent"]) > 1e-3 or abs(dice - s["mean_dice_percent"]) > 1e-3:
            fail(f"{name}: IoU/Dice in the summary differ from the per-image file")
        k = acc = None
        if s.get("accuracy_percent") is not None:
            k = int(pd.to_numeric(p["correct"]).sum())
            acc = 100 * k / n
            if k != s["n_correct"] or abs(acc - s["accuracy_percent"]) > 1e-6:
                fail(f"{name}: accuracy in the summary differs from {k}/{n}")
        fp_normal = int(((~pos) & (p["fp"].astype(float) > 0)).sum()) if n_neg else None
        rows.append({"dataset": ds, "config": cfg, "seed": seed, "iou": iou, "dice": dice, "k": k, "n": n,
                     "acc": acc, "auc": s.get("macro_ovr_auc_percent"), "fp_normal": fp_normal,
                     "fp_area": s.get("empty_mask_fg_fraction_percent")})
        print(f"ok   {name:44s} IoU {iou:6.2f} Dice {dice:6.2f} "
              + (f"Acc {k}/{n} = {acc:6.2f} " if k is not None else "Acc  -  ")
              + (f"AUC {s['macro_ovr_auc_percent']:6.2f}" if s.get("macro_ovr_auc_percent") is not None else "")
              + (f" | normal images with lesion pixels {fp_normal}/{n_neg}" if fp_normal is not None else ""))

    df = pd.DataFrame(rows)
    if df.empty:
        fail("no runs found")
    else:
        print("\n=== values for the tables (mean and sample std over seeds; differences use the unrounded means) ===")
        for (ds, cfg), g in df.groupby(["dataset", "config"]):
            line = f"{ds:15s} {cfg:30s} seeds {sorted(g.seed.tolist())}"
            for col in ("iou", "dice", "acc", "auc"):
                v = g[col].dropna().astype(float)
                if len(v):
                    line += f" | {col} {v.mean():.2f}" + (f" ± {v.std(ddof=1):.2f}" if len(v) > 1 else "")
                    line += f" -> {v.mean():.1f}"
            if g["k"].notna().all():
                line += f" | acc = {int(g.k.sum())}/{int(g.n.sum())}"
            if g["fp_normal"].notna().all():
                line += f" | normal with lesion pixels {int(g.fp_normal.sum())}/{21 * len(g)}"
            print(line)

        print("\n=== missing runs ===")
        have = {(r.dataset, r.config, r.seed) for r in df.itertuples()}
        need = []
        for ds in ("busi", "busi_whu"):
            for cfg in ("full", ABLATION[1]):
                need += [(ds, cfg, s) for s in (42, 43, 44)]
            need += [(ds, c, 42) for c in COMPARISON]
        need += [("busi", ABLATION[r], s) for r in (2, 3, 4, 6, 7, 8) for s in (42, 43, 44)]
        need += [("busi_whu->busi", "full", s) for s in (42, 43, 44)]
        missing = [x for x in need if x not in have]
        for x in missing:
            fail(f"missing run {x[0]}/{x[1]}/s{x[2]}")
        if not missing:
            print("none")
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} problem(s) above'}")


if __name__ == "__main__":
    main()
