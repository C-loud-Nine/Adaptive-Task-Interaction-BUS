"""
train.py - train, select and test one run of the proposed model (PyTorch).

    python train.py --dataset busi --seed 42                          # full model
    python train.py --dataset busi --mscf off --tim off               # baseline (Table 2, row 1)
    python train.py --dataset busi --aiw-mode sigmoid                 # Table 2, row 8

Protocol (Sec. 3.1): mixed precision, Adam (beta1 0.91), batch 24, gradient
clipping at 1, up to 100 epochs with cosine decay and plateau reduction.
Epochs 1-3 (at most 100 batches each) are not used for selection; from epoch 4
on, AIW is calibrated on the un-augmented training images after every epoch and
the validation score 0.7 * pooled foreground IoU + 0.3 * accuracy selects the
model (best epoch restored; early stopping after 22 epochs without a gain of
0.004). The restored model is tested once; BUSI-WHU runs are also tested on the
BUSI lesion images (external test).

Everything a run produces goes to --out (default runs/<dataset>_<variant>_s<seed>):
results/ (split manifest, exclusions, per-image predictions, summaries,
validation history), checkpoints/, train_log.txt and complexity_log.txt, the
layout read by analysis/aggregate.py. A training state is saved after every
epoch, so an interrupted run continues where it stopped.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import config
from data_loader import prepare_datasets, external_busi_lesions
from metrics import write_outputs
from model import (DEVICE, AdaptiveInteractionModel, CosinePlateau, calibrate_aiw, clf_loss, predict,
                   seg_loss, to_tensor)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="busi", choices=["busi", "busi_whu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mscf", default="on", choices=["on", "off"])
    ap.add_argument("--tim", default="on", choices=["on", "off"])
    ap.add_argument("--aiw-mode", default="adaptive", choices=["adaptive", "off", "fixed", "sigmoid", "constant"])
    ap.add_argument("--tim-direction", default="both", choices=["both", "seg2clf", "clf2seg"])
    ap.add_argument("--max-epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--out", default=None, help="run folder (default runs/<dataset>_<variant>_s<seed>)")
    return ap.parse_args()


def main():
    args = parse_args()
    config.EPOCHS = args.max_epochs
    variant = ("full" if (args.mscf, args.tim, args.aiw_mode, args.tim_direction) == ("on", "on", "adaptive", "both")
               else f"mscf-{args.mscf}_tim-{args.tim}_aiw-{args.aiw_mode}"
               + (f"_dir-{args.tim_direction}" if args.tim_direction != "both" else ""))
    out = args.out or os.path.join("runs", f"{args.dataset}_{variant}_s{args.seed}")
    config.RESULTS_DIR = os.path.join(out, "results")
    config.CHECKPOINT_DIR = os.path.join(out, "checkpoints")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    data_path = config.BUSI_WHU_PATH if args.dataset == "busi_whu" else config.BUSI_PATH
    classes = config.WHU_CLASSES if args.dataset == "busi_whu" else config.CATEGORIES

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    LOG = []

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        LOG.append(line)

    log(f"Run: dataset {args.dataset} | seed {args.seed} | variant {variant} | device {DEVICE} | output {out}")

    # ---- data: curation, duplicate grouping, split, augmentation ------------------------------
    data = prepare_datasets(data_path, dataset=args.dataset)
    LOG.append(f"Duplicate grouping: {data['dedup_report']}")
    man = pd.read_csv(os.path.join(config.RESULTS_DIR, f"split_manifest_{args.dataset}.csv"))
    assert (man.groupby("group")["split"].nunique() == 1).all(), "a duplicate group crosses partitions"
    print(man.groupby(["split", "class"]).size().unstack(fill_value=0).to_string())

    # ---- model, optimiser, schedule ----------------------------------------------------------------
    def build(pretrained=True):
        return AdaptiveInteractionModel(num_classes=len(classes), mscf=args.mscf == "on", tim=args.tim == "on",
                                        aiw=args.aiw_mode, direction=args.tim_direction,
                                        dropout=config.DROPOUT_RATE, pretrained=pretrained).to(DEVICE)

    model = build()
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,} (variant {variant})")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.INITIAL_LR,
                           betas=(config.ADAM_BETA_1, config.ADAM_BETA_2), eps=config.ADAM_EPSILON)
    sched = CosinePlateau(opt)
    amp = bool(config.MIXED_PRECISION) and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    Xtr = data["X_train"]
    Mtr = torch.from_numpy(np.ascontiguousarray(data["mask_train"])).permute(0, 3, 1, 2).float()
    Ytr = torch.from_numpy(np.asarray(data["y_train"]).astype(np.int64))
    m_val = (np.asarray(data["mask_val"])[..., 0] > 0.5).reshape(len(data["X_val"]), -1)
    y_val = np.asarray(data["y_val"]).astype(int)
    gen = torch.Generator().manual_seed(args.seed)
    steps = int(np.ceil(len(Xtr) / config.BATCH_SIZE))

    def run_epoch(epoch, max_batches=None):
        model.train(); sched.begin(epoch)
        order = torch.randperm(len(Xtr), generator=gen).numpy()
        nb = min(steps, max_batches or steps)
        tot = seg_t = clf_t = 0.0
        bar = tqdm(range(nb), desc=f"Epoch {epoch + 1}", unit="batch", leave=False)
        for b in bar:
            idx = order[b * config.BATCH_SIZE:(b + 1) * config.BATCH_SIZE]
            xb, mb, yb = to_tensor(Xtr[idx]).to(DEVICE, non_blocking=True), Mtr[idx].to(DEVICE), Ytr[idx].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                o = model(xb)
            ls, lc = seg_loss(mb, o["seg"].float()), clf_loss(yb, o["clf"].float())
            loss = config.SEG_WEIGHT * ls + config.CLF_WEIGHT * lc
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.GLOBAL_CLIPNORM)
            scaler.step(opt); scaler.update()
            tot += loss.item(); seg_t += ls.item(); clf_t += lc.item()
            bar.set_postfix(loss=f"{tot / (b + 1):.4f}")
        return tot / nb, seg_t / nb, clf_t / nb

    def val_score():
        calibrate_aiw(model, data["X_train_orig"], config.VAL_BATCH_SIZE)
        seg, clf, _ = predict(model, data["X_val"], config.VAL_BATCH_SIZE)
        p = seg.reshape(len(seg), -1) > config.SEG_THRESHOLD
        tp = np.logical_and(p, m_val).sum(); fp = np.logical_and(p, ~m_val).sum(); fn = np.logical_and(~p, m_val).sum()
        j = tp / max(tp + fp + fn, 1)
        a = float((clf.argmax(1) == y_val).mean())
        return config.SELECTION_IOU_WEIGHT * j + (1 - config.SELECTION_IOU_WEIGHT) * a, j, a

    # ---- training with automatic resume ------------------------------------------------------------
    best, best_es, best_epoch, wait, best_state, history, stopped, start = -np.inf, -np.inf, -1, 0, None, [], False, 0
    resume = os.path.join(out, "resume.pt")
    if os.path.exists(resume):
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        for k in ("best", "wait", "cool", "factor"):
            setattr(sched, k, st["sched"][k])
        best, best_es, best_epoch, wait, best_state, history, stopped = (st[k] for k in (
            "best", "best_es", "best_epoch", "wait", "best_state", "history", "stopped"))
        gen.set_state(st["gen"]); torch.set_rng_state(st["cpu_rng"])
        np.random.set_state(st["np_rng"]); random.setstate(st["py_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        start = st["epoch"]
        print(f"Resumed from {resume}: {start} epochs done, best epoch {best_epoch}")

    def save_resume(epoch_done):
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
              "sched": {k: getattr(sched, k) for k in ("best", "wait", "cool", "factor")},
              "best": best, "best_es": best_es, "best_epoch": best_epoch, "wait": wait, "best_state": best_state,
              "history": history, "stopped": stopped, "epoch": epoch_done, "gen": gen.get_state(),
              "cpu_rng": torch.get_rng_state(), "np_rng": np.random.get_state(), "py_rng": random.getstate(),
              "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(st, resume + ".tmp")
        os.replace(resume + ".tmp", resume)

    for e in range(start, config.EPOCHS if not stopped else start):
        t = time.time()
        loss, ls, lc = run_epoch(e, max_batches=100 if e < config.WARMUP_EPOCHS else None)
        lr = opt.param_groups[0]["lr"]
        if e < config.WARMUP_EPOCHS:
            print(f"Epoch {e + 1:3d}/{config.EPOCHS} {time.time() - t:5.0f}s | loss {loss:.4f} "
                  f"(seg {ls:.4f}, clf {lc:.4f}) | lr {lr:.2e} | not used for selection")
            save_resume(e + 1)
            continue
        s, j, a = val_score()
        history.append((e + 1, s, j, a, loss, lr))
        sched.end(s)
        if s > best:
            best, best_epoch = s, e + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if s > best_es + config.EARLY_STOPPING_MIN_DELTA:
            best_es, wait = s, 0
        else:
            wait += 1
        print(f"Epoch {e + 1:3d}/{config.EPOCHS} {time.time() - t:5.0f}s | loss {loss:.4f} | val score {s:.4f} "
              f"IoU {j:.4f} acc {a:.4f} | lr {lr:.2e} | best {best:.4f} @ {best_epoch} "
              f"({wait}/{config.EARLY_STOPPING_PATIENCE})")
        if wait >= config.EARLY_STOPPING_PATIENCE:
            stopped = True
            print(f"Early stopping at epoch {e + 1}; best epoch {best_epoch}")
        save_resume(e + 1)
        if stopped:
            break

    model.load_state_dict(best_state)
    for name, (ms, mc) in calibrate_aiw(model, data["X_train_orig"], config.VAL_BATCH_SIZE).items():
        log(f"AIW calibration {name}: m_seg = {ms:.6g}, m_clf = {mc:.6g}")
    ckpt = os.path.join(config.CHECKPOINT_DIR, f"best_{args.dataset}_{variant}_s{args.seed}.pt")
    torch.save(model.state_dict(), ckpt)
    with open(os.path.join(config.RESULTS_DIR, f"val_history_{args.dataset}_{variant}_s{args.seed}.csv"), "w") as fh:
        fh.write("epoch,val_score,val_fg_iou,val_acc,train_loss,lr\n")
        for row in history:
            fh.write(",".join(str(v) for v in row) + "\n")
    ledger = os.path.join(config.RESULTS_DIR, "run_ledger.csv")
    new = not os.path.exists(ledger)
    with open(ledger, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "dataset", "training_seed", "split_seed", "dedup", "variant", "best_epoch",
                        "best_val_score", "stopped_early", "checkpoint", "checkpoint_sha256_16"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), args.dataset, args.seed,
                    config.RANDOM_STATE, config.DEDUP, variant, best_epoch, f"{best:.6f}", int(stopped), ckpt,
                    hashlib.sha256(open(ckpt, "rb").read()).hexdigest()[:16]])
    log(f"Model saved to {ckpt}; best epoch {best_epoch} (val_score {best:.4f}); stopped early: {stopped}")

    # ---- test: the saved checkpoint is reloaded -------------------------------------------------------
    test_model = build(pretrained=False)
    test_model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    tag = f"{os.path.splitext(os.path.basename(ckpt))[0]}_bs{config.VAL_BATCH_SIZE}"
    seg, clf, ws = predict(test_model, data["X_test"], config.VAL_BATCH_SIZE)
    extra = {f"w_{k}_d{l + 1}": wl[:, i] for l, wl in enumerate(ws) for i, k in enumerate(("seg", "clf"))}
    summ = write_outputs(tag, args.dataset, ckpt, config.VAL_BATCH_SIZE, data["paths_test"],
                         np.asarray(data["y_test"]).astype(int), clf, seg, data["mask_test"][..., 0], classes,
                         config.RESULTS_DIR, extra=extra, save_masks=True)
    summ["mean_aiw_weights"] = {f"d{l + 1}": {"w_seg": float(wl[:, 0].mean()), "w_clf": float(wl[:, 1].mean())}
                                for l, wl in enumerate(ws)}
    json.dump(summ, open(os.path.join(config.RESULTS_DIR, f"summary_{tag}.json"), "w"), indent=2)
    log(f"Test: IoU {summ['mean_iou_percent']:.2f}% | Dice {summ['mean_dice_percent']:.2f}% | "
        f"Acc {summ['accuracy_percent']:.2f}% ({summ['n_correct']}/{summ['n_test_images']})")

    if args.dataset == "busi_whu":           # external test: curated BUSI lesion images, BUSI-WHU copies removed
        ext = external_busi_lesions(config.BUSI_PATH, config.BUSI_WHU_PATH)
        msg = (f"[external] {len(ext['X'])} BUSI lesion images; "
               f"{ext['n_cross_dataset_duplicates']} removed as copies of BUSI-WHU images")
        print(msg)
        open(os.path.join(out, "eval_ext_log.txt"), "w").write(msg + "\n")
        seg_e, clf_e, _ = predict(test_model, ext["X"], config.VAL_BATCH_SIZE)
        e_s = write_outputs(f"ext-busi_{os.path.splitext(os.path.basename(ckpt))[0]}_bs{config.VAL_BATCH_SIZE}",
                            "busi_whu->busi", ckpt, config.VAL_BATCH_SIZE, ext["paths"], ext["y"], clf_e, seg_e,
                            ext["mask"][..., 0], config.WHU_CLASSES, config.RESULTS_DIR)
        log(f"External test: IoU {e_s['mean_iou_percent']:.2f}% | Acc {e_s['accuracy_percent']:.2f}%")

    # ---- complexity ---------------------------------------------------------------------------------
    from torch.utils.flop_counter import FlopCounterMode
    test_model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        test_model(torch.zeros(1, 3, 224, 224, device=DEVICE))
    cx = [f"parameters: total {n_params:,} | trainable {n_train:,}",
          f"FLOPs per 224x224 image: {fc.get_total_flops() / 1e9:.2f} G"]
    print(*cx, sep="\n")
    open(os.path.join(out, "complexity_log.txt"), "w").write("\n".join(cx) + "\n")
    open(os.path.join(out, "train_log.txt"), "w").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
