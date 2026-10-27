"""
train_baseline.py - train and test one comparison method under the protocol of
the proposed model (same curated split, augmentation, loss, optimiser,
schedule, model selection and metrics). PyTorch.

Usage:
    python baselines/train_baseline.py --model mti_net --dataset busi --seed 42

Shared with the proposed model (imported from the repository root):
  data_loader.prepare_datasets   curation, duplicate grouping, split, augmentation
  metrics.write_outputs          test metrics and result files
Re-implemented here as in the proposed model's code (model.py, train.py):
  segmentation loss (loss.enhanced_lesion_focus_loss), focal cross-entropy,
  Adam (beta1 0.91, beta2 0.999, eps 1e-7), global gradient-norm clipping at 1,
  cosine decay with plateau reduction (train.CosineDecayScheduler),
  mixed precision, epochs 1-3 (at most 100 batches each) not used for selection, validation selection
  0.7 * pooled foreground IoU + 0.3 * accuracy (IoU only for segmentation-only
  models), best-epoch restore, early stopping (patience 22, min delta 0.004).
"""
import argparse
import copy
import csv
import datetime
import hashlib
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                      # noqa: E402
import data_loader                 # noqa: E402
import metrics                     # noqa: E402
from models import build_model, MODEL_INFO   # noqa: E402



# ---------------------------------------------------------------------------
# losses (identical to model.py)
# ---------------------------------------------------------------------------
def seg_loss(y_true, y_pred):
    eps = 1e-7
    y_true = y_true.clamp(0, 1)
    y_pred = torch.nan_to_num(y_pred, 0.0).clamp(eps, 1 - eps)
    dims = (1, 2, 3)
    tp = (y_true * y_pred).sum(dims)
    fp = ((1 - y_true) * y_pred).sum(dims)
    fn = (y_true * (1 - y_pred)).sum(dims)
    t_idx = (tp + eps) / (tp + 0.5 * fp + 0.5 * fn + eps)
    focal = torch.clamp(1 - t_idx, min=eps) ** 0.75
    ratio = y_true.sum(dims) / y_true[0].numel()
    size_w = 1 + torch.exp(-10 * ratio)
    pos = (ratio > 0).float()
    ft = (pos * size_w * focal).sum() / pos.sum().clamp(min=1)
    bce = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()
    sob = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=y_pred.device).view(1, 1, 3, 3)
    gt = F.conv2d(y_true, sob, padding=1)
    gp = F.conv2d(y_pred, sob, padding=1)
    std_gt = torch.sqrt(gt.var(dim=dims, unbiased=False).clamp(min=0) + 1e-8)
    std_gp = torch.sqrt(gp.var(dim=dims, unbiased=False).clamp(min=0) + 1e-8)
    edge = (std_gt - std_gp).abs().mean()
    return ft + config.BCE_WEIGHT * bce + config.EDGE_WEIGHT * edge


def clf_loss(y_true, prob):
    eps = 1e-7
    prob = torch.nan_to_num(prob, eps).clamp(eps, 1 - eps)
    prob = prob / prob.sum(1, keepdim=True)
    prob = prob.clamp(eps, 1 - eps)
    ce = -torch.log(prob.gather(1, y_true.view(-1, 1)).squeeze(1))
    pt = torch.exp(-ce)
    return (torch.clamp(1 - pt, min=eps) ** 2 * ce).clamp(0, 10).mean()


def to_float32(o):
    """Cast model outputs (tensors, dicts, lists, tuples) to float32."""
    if torch.is_tensor(o):
        return o.float()
    if isinstance(o, dict):
        return {k: to_float32(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(to_float32(v) if torch.is_tensor(v) or isinstance(v, (dict, list, tuple)) else v for v in o)
    return o


def amp_enabled(device):
    """Mixed precision (float16 autocast + GradScaler) on CUDA only, as for the proposed model."""
    return bool(config.MIXED_PRECISION) and str(device).startswith("cuda")


def total_loss(out, mask, label, seg_only):
    """0.82 L_seg + 0.18 L_clf on the main outputs, plus the weighted
    deep-supervision terms of methods whose original training defines them."""
    def one(o):
        l = config.SEG_WEIGHT * seg_loss(mask, o["seg"])
        if not seg_only and o.get("clf") is not None:
            l = l + config.CLF_WEIGHT * clf_loss(label, o["clf"])
        return l
    return one(out) + sum(w * one(a) for a, w in out.get("aux", []))


# ---------------------------------------------------------------------------
# learning-rate schedule (identical to train.CosineDecayScheduler)
# ---------------------------------------------------------------------------
class CosinePlateau:
    def __init__(self, opt):
        self.opt, self.best, self.wait, self.cool, self.factor = opt, -np.inf, 0, 0, 1.0

    def lr(self, epoch):
        prog = epoch / float(config.EPOCHS)
        cos = 0.5 * (1 + np.cos(np.pi * prog))
        base = config.INITIAL_LR * (config.COSINE_ALPHA + (1 - config.COSINE_ALPHA) * cos)
        return max(base * self.factor, config.MIN_LR)

    def begin(self, epoch):
        for g in self.opt.param_groups:
            g["lr"] = self.lr(epoch)

    def end(self, score):
        if score is None:
            return
        if self.cool > 0:
            self.cool -= 1
            return
        if score > self.best:
            self.best, self.wait = score, 0
        else:
            self.wait += 1
            if self.wait >= config.REDUCE_LR_PATIENCE:
                self.factor = max(self.factor * config.REDUCE_LR_FACTOR, config.MIN_LR / config.INITIAL_LR)
                self.cool, self.wait = config.REDUCE_LR_COOLDOWN, 0


# ---------------------------------------------------------------------------
def to_tensor(x, norm):
    """[N,H,W,3] in [0,255] -> [N,3,H,W] in [0,1], normalised with the backbone's mean/std (norm)."""
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float() / 255.0
    if norm is not None:
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in norm)
        t = (t - mean) / std
    return t


@torch.no_grad()
def predict(model, X, norm, device, bs):
    model.eval()
    segs, clfs = [], []
    for i in range(0, len(X), bs):
        xb = to_tensor(X[i:i + bs], norm).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled(device)):
            o = model(xb)
        segs.append(o["seg"].float().cpu())
        if o.get("clf") is not None:
            clfs.append(o["clf"].float().cpu())
    seg = torch.cat(segs).numpy()[:, 0]
    clf = torch.cat(clfs).numpy() if clfs else None
    return seg, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_INFO))
    ap.add_argument("--dataset", default="busi", choices=["busi", "busi_whu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--out", default=None,
                    help="run folder for results/, checkpoints/ and train_log.txt (default: ./results, ./checkpoints)")
    args = ap.parse_args()
    config.EPOCHS = args.max_epochs
    if args.out:                                 # one folder per run, the layout read by analysis/aggregate.py
        config.RESULTS_DIR = os.path.join(args.out, "results")
        config.CHECKPOINT_DIR = os.path.join(args.out, "checkpoints")
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, x):
                for st in self.streams:
                    st.write(x); st.flush()
            def flush(self):
                for st in self.streams:
                    st.flush()
        sys.stdout = _Tee(sys.__stdout__, open(os.path.join(args.out, "train_log.txt"), "w"))

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    info = MODEL_INFO[args.model]
    seg_only, pretrained, norm = info["seg_only"], info["pretrained"], info["norm"]
    print("=" * 80)
    print(f"Baseline {args.model} | {args.dataset} | seed {args.seed} | seg_only {seg_only} "
          f"| pretrained {pretrained} | device {device}")
    print("=" * 80)

    path = config.BUSI_WHU_PATH if args.dataset == "busi_whu" else config.BUSI_PATH
    if args.dataset == "busi_whu":           # the external test needs BUSI: fail now, not after training
        for c in config.CATEGORIES:
            assert os.path.isdir(os.path.join(config.BUSI_PATH, c)), f"BUSI not found at {config.BUSI_PATH} (external test)"
    cls = config.WHU_CLASSES if args.dataset == "busi_whu" else config.CATEGORIES
    data = data_loader.prepare_datasets(path, dataset=args.dataset)
    print("Duplicate grouping:", data["dedup_report"])

    model = build_model(args.model, num_classes=len(cls)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    gflops = None
    try:                                         # FLOPs of one 224 x 224 image (multiply-add = 2)
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(torch.zeros(1, 3, 224, 224, device=device))
        gflops = fc.get_total_flops() / 1e9
    except Exception as exc:                     # older PyTorch
        print("FLOP count unavailable:", exc)
    print(f"parameters: {n_params:,} | GFLOPs per image: {gflops}")
    opt = torch.optim.Adam(model.parameters(), lr=config.INITIAL_LR,
                           betas=(config.ADAM_BETA_1, config.ADAM_BETA_2), eps=config.ADAM_EPSILON)
    sched = CosinePlateau(opt)
    # mixed precision (float16 compute, dynamic loss scaling) as for the proposed model;
    # the loss is computed in float32
    scaler = (torch.amp.GradScaler("cuda", enabled=amp_enabled(device)) if hasattr(torch, "amp")
              and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=amp_enabled(device)))

    Xtr = data["X_train"]
    Mtr = torch.from_numpy(np.ascontiguousarray(data["mask_train"])).permute(0, 3, 1, 2).float()
    Ytr = torch.from_numpy(np.asarray(data["y_train"]).astype(np.int64))
    m_val = (np.asarray(data["mask_val"])[..., 0] > 0.5).reshape(len(data["X_val"]), -1)
    y_val = np.asarray(data["y_val"]).astype(int)
    g = torch.Generator().manual_seed(args.seed)

    def run_epoch(epoch, max_batches=None):
        model.train()
        sched.begin(epoch)
        order = torch.randperm(len(Xtr), generator=g).numpy()
        nb = (len(order) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
        if max_batches:
            nb = min(nb, max_batches)
        tot = 0.0
        for b in range(nb):
            idx = order[b * config.BATCH_SIZE:(b + 1) * config.BATCH_SIZE]
            xb = to_tensor(Xtr[idx], norm).to(device)
            mb, yb = Mtr[idx].to(device), Ytr[idx].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled(device)):
                out = model(xb)
            loss = total_loss(to_float32(out), mb, yb, seg_only)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GLOBAL_CLIPNORM)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        return tot / max(nb, 1)

    def val_score():
        seg, clf = predict(model, data["X_val"], norm, device, config.VAL_BATCH_SIZE)
        p = seg.reshape(len(seg), -1) > config.SEG_THRESHOLD
        tp = np.logical_and(p, m_val).sum(); fp = np.logical_and(p, ~m_val).sum()
        fn = np.logical_and(~p, m_val).sum()
        j = tp / max(tp + fp + fn, 1)
        if seg_only:
            return j, j, None
        a = float((clf.argmax(1) == y_val).mean())
        w = config.SELECTION_IOU_WEIGHT
        return w * j + (1 - w) * a, j, a

    for e in range(config.WARMUP_EPOCHS):
        l = run_epoch(e, max_batches=100)
        print(f"warm-up epoch {e+1}: loss {l:.4f}")

    best, best_es, best_epoch, wait, best_state, history = -np.inf, -np.inf, -1, 0, None, []
    stopped = False
    try:
        from tqdm.auto import tqdm
        epochs = tqdm(range(config.WARMUP_EPOCHS, config.EPOCHS), desc="epochs")
    except ImportError:
        epochs = range(config.WARMUP_EPOCHS, config.EPOCHS)
    for e in epochs:
        l = run_epoch(e)
        s, j, a = val_score()
        history.append((e + 1, s, j, a if a is not None else ""))
        print(f"epoch {e+1}: loss {l:.4f} | val_score {s:.4f} (J_val {j:.4f}"
              + (f", A_val {a:.4f})" if a is not None else ")"), flush=True)
        sched.end(s)
        if s > best:
            best, best_epoch = s, e + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if s > best_es + config.EARLY_STOPPING_MIN_DELTA:
            best_es, wait = s, 0
        else:
            wait += 1
            if wait >= config.EARLY_STOPPING_PATIENCE:
                stopped = True
                print(f"early stopping at epoch {e+1}; best epoch {best_epoch}")
                break
    model.load_state_dict(best_state)
    print(f"restored epoch {best_epoch} (val_score {best:.4f})")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    stem = f"best_{args.dataset}_{args.model}_s{args.seed}"
    ckpt = os.path.join(config.CHECKPOINT_DIR, stem + ".pt")
    torch.save(best_state, ckpt)
    with open(os.path.join(config.RESULTS_DIR, f"val_history_{args.dataset}_{args.model}_s{args.seed}.csv"), "w") as fh:
        fh.write("epoch,val_score,val_fg_iou,val_acc\n")
        for row in history:
            fh.write(",".join(str(v) for v in row) + "\n")

    seg, clf = predict(model, data["X_test"], norm, device, config.VAL_BATCH_SIZE)
    metrics.write_outputs(f"{stem}_bs{config.VAL_BATCH_SIZE}", args.dataset, ckpt, config.VAL_BATCH_SIZE,
                          data["paths_test"], data["y_test"], clf, seg, data["mask_test"], cls,
                          config.RESULTS_DIR, save_masks=True)
    if args.dataset == "busi_whu":               # external test: all curated BUSI lesion images
        ext = data_loader.external_busi_lesions(config.BUSI_PATH, config.BUSI_WHU_PATH)
        print(f"external: {len(ext['X'])} BUSI lesion images; "
              f"{ext['n_cross_dataset_duplicates']} removed as copies of BUSI-WHU images")
        seg_e, clf_e = predict(model, ext["X"], norm, device, config.VAL_BATCH_SIZE)
        metrics.write_outputs(f"ext-busi_{stem}_bs{config.VAL_BATCH_SIZE}", "busi_whu->busi", ckpt,
                              config.VAL_BATCH_SIZE, ext["paths"], ext["y"], clf_e, seg_e, ext["mask"],
                              config.WHU_CLASSES, config.RESULTS_DIR)

    with open(ckpt, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()[:16]
    ledger = os.path.join(config.RESULTS_DIR, "run_ledger.csv")
    new = not os.path.exists(ledger)
    with open(ledger, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "dataset", "training_seed", "split_seed", "dedup", "variant",
                        "input_scale", "best_epoch", "best_val_score", "stopped_early",
                        "checkpoint", "checkpoint_sha256_16"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), args.dataset, args.seed,
                    config.RANDOM_STATE, config.DEDUP, args.model,
                    "unit" if norm is None else f"mean{norm[0]} std{norm[1]}", best_epoch, f"{best:.6f}", int(stopped),
                    ckpt, sha])
    print(f"parameters: {n_params:,} | GFLOPs per image: {gflops} | best epoch {best_epoch} "
          f"| stopped early {stopped}")


if __name__ == "__main__":
    main()
