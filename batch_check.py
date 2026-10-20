"""
batch_check.py - does the evaluation batch size change the predictions?

AIW divides each dispersion statistic by its batch mean, so a prediction can
depend on the other images in the batch. This script predicts the same test
images, in the same order, at several batch sizes and reports per-image
differences in class probabilities, segmentation masks and AIW weights.

Usage:
    python batch_check.py --checkpoint checkpoints/best_busi_adaptive_s42.h5 --sizes 1 8 24
Writes results/batch_check_<checkpoint>.csv and prints the largest changes.
"""
import argparse, os
import numpy as np
import tensorflow as tf

import config, data_loader
from evaluate import CUSTOM_OBJECTS, build_weight_export_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="busi", choices=["busi", "busi_whu"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 24])
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.checkpoint, compile=False,
                                       custom_objects=CUSTOM_OBJECTS)
    path = config.BUSI_WHU_PATH if args.dataset == "busi_whu" else config.BUSI_PATH
    data = data_loader.prepare_datasets(path, dataset=args.dataset)
    X, files = data["X_test"], data["paths_test"]
    export, levels = build_weight_export_model(model)

    runs = {}
    for b in args.sizes:
        out = export.predict(X, batch_size=b, verbose=0)
        seg = (np.asarray(out[0])[..., 0] > 0.5)
        clf = np.asarray(out[1])
        w = np.concatenate([np.asarray(o).reshape(len(X), -1) for o in out[2:]], 1) if levels else None
        runs[b] = (seg, clf, w)

    ref = args.sizes[0]
    rows = []
    for b in args.sizes[1:]:
        s0, c0, w0 = runs[ref]; s1, c1, w1 = runs[b]
        for i, f in enumerate(files):
            rows.append([f, ref, b,
                         float(np.abs(c1[i] - c0[i]).max()),
                         int(c1[i].argmax() != c0[i].argmax()),
                         float((s1[i] != s0[i]).mean()),
                         float(np.abs(w1[i] - w0[i]).max()) if w0 is not None else np.nan])
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = os.path.join(config.RESULTS_DIR, f"batch_check_{stem}.csv")
    with open(out, "w") as fh:
        fh.write("file,size_a,size_b,max_prob_change,label_changed,mask_pixels_changed,max_aiw_change\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    for b in args.sizes[1:]:
        sel = [r for r in rows if r[2] == b]
        print(f"batch {ref} vs {b}: labels changed {sum(r[4] for r in sel)}/{len(sel)}, "
              f"max prob change {max(r[3] for r in sel):.4f}, "
              f"max mask pixel fraction changed {max(r[5] for r in sel):.4f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
