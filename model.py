"""The proposed model.

Modules (MSCF, encoder attention, residual block, attention gate, TIM, AIW), the
network with the EfficientNet-B4 encoder, AIW calibration, prediction, losses and
the learning-rate schedule. Training and testing: train.py.
"""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def bn_layer(c, d=2):
    """Batch normalisation with momentum 0.01 (PyTorch convention; running stats decay 0.99) and eps 1e-3."""
    return (nn.BatchNorm2d if d == 2 else nn.BatchNorm1d)(c, eps=1e-3, momentum=0.01)


class SepConv(nn.Module):
    """Separable convolution: 3x3 depthwise (no bias) + 1x1 pointwise (bias), 'same' padding."""
    def __init__(self, cin, cout, dilation=1, relu=False):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=dilation, dilation=dilation, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1)
        self.relu = relu

    def forward(self, x):
        x = self.pw(self.dw(x))
        return F.relu(x) if self.relu else x


class MultiScaleContextFusion(nn.Module):
    """MSCF (Eqs. 9-10): dilated separable convolutions (rates 1, 2, 4) fused by a per-image softmax over scales."""
    def __init__(self, c, reduction=8):
        super().__init__()
        self.branches = nn.ModuleList([SepConv(c, c, d) for d in (1, 2, 4)])
        self.att = nn.Sequential(nn.Linear(c, c // reduction), nn.ReLU(), nn.Linear(c // reduction, 3))
        self.fusion = nn.Conv2d(c, c, 1)

    def forward(self, x):
        w = torch.softmax(self.att(x.mean((2, 3))), dim=1)                      # [B, 3]
        y = sum(b(x) * w[:, i, None, None, None] for i, b in enumerate(self.branches))
        return self.fusion(y)


class DualPathAttention(nn.Module):
    """CBAM-style channel and spatial attention (encoder blocks)."""
    def __init__(self, c):
        super().__init__()
        self.channel = nn.Sequential(nn.Linear(c, max(8, c // 16)), nn.ReLU(), nn.Linear(max(8, c // 16), c), nn.Sigmoid())
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        cw = self.channel(x.mean((2, 3)))[:, :, None, None]
        sw = torch.sigmoid(self.spatial(torch.cat([x.amax(1, keepdim=True), x.mean(1, keepdim=True)], 1)))
        return x * cw * sw


class ResidualBlock(nn.Module):
    """Two separable convolutions with BN, MSCF, optional attention, residual (1x1 projection if needed)."""
    def __init__(self, cin, c, attention=True, mscf=True):
        super().__init__()
        self.conv1, self.bn1 = SepConv(cin, c, relu=True), bn_layer(c)
        self.conv2, self.bn2 = SepConv(c, c), bn_layer(c)
        self.mscf = MultiScaleContextFusion(c) if mscf else None
        self.att = DualPathAttention(c) if attention else None
        self.proj = nn.Conv2d(cin, c, 1) if cin != c else None

    def forward(self, x):
        y = self.bn2(self.conv2(self.bn1(self.conv1(x))))
        if self.mscf is not None:
            y = self.mscf(y)
        if self.att is not None:
            y = self.att(y)
        return F.relu(y + (self.proj(x) if self.proj is not None else x))


class AttentionGate(nn.Module):
    """Attention gate with a dilated context branch on the skip connection."""
    def __init__(self, x_ch, g_ch, inter):
        super().__init__()
        self.tx, self.tg = nn.Conv2d(x_ch, inter, 1), nn.Conv2d(g_ch, inter, 1)
        self.context = nn.Sequential(SepConv(inter, inter, 1), SepConv(inter, inter, 2), nn.Conv2d(inter, 1, 1), nn.Sigmoid())

    def forward(self, x, g):
        return x * self.context(F.relu(self.tx(x) + self.tg(g)))


class TaskInteractionModule(nn.Module):
    """TIM (Eqs. 1-3). Both directions read the pre-interaction features."""
    def __init__(self, seg_c, clf_c=256, direction="both", tau=0.7):
        super().__init__()
        self.direction, self.tau = direction, tau
        if direction in ("both", "seg2clf"):
            self.s_conv = nn.Conv2d(seg_c, seg_c // 2, 1)
            self.s_fc, self.s_bn = nn.Linear(seg_c // 2, clf_c), bn_layer(clf_c, 1)
            self.clf_gate = nn.Linear(clf_c, clf_c)
        if direction in ("both", "clf2seg"):
            self.p_fc, self.g_fc = nn.Linear(clf_c, seg_c), nn.Linear(seg_c, seg_c)

    def forward(self, D, f):
        f_enh, D_enh = f, D
        if self.direction in ("both", "seg2clf"):
            s = self.s_bn(F.relu(self.s_fc(F.relu(self.s_conv(D)).mean((2, 3)))))        # Eq. 1
            f_enh = f + torch.sigmoid(self.clf_gate(s)) * s                              # Eq. 2
        if self.direction in ("both", "clf2seg"):
            p = torch.sigmoid(self.p_fc(f))[:, :, None, None]
            g = torch.sigmoid(self.g_fc(D.mean((2, 3))))[:, :, None, None]
            D_enh = D * (1.0 + self.tau * g * p)                                          # Eq. 3
        return D_enh, f_enh


def activation_dispersion(x):
    """Eq. 4: spatial variance averaged over channels (maps) or variance (vectors), float32."""
    x = x.float()
    if x.dim() == 4:
        return x.var(dim=(2, 3), unbiased=False).mean(1, keepdim=True)
    return x.var(dim=1, unbiased=False, keepdim=True)


class AdaptiveInteractionWeighting(nn.Module):
    """AIW (Eqs. 4-8), computed in float32. m_t: batch mean in training; fixed calibrated mean at inference."""
    def __init__(self, mode="adaptive"):
        super().__init__()
        self.mode = mode
        if mode in ("adaptive", "sigmoid"):
            self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
        if mode == "constant":                                # learned per-level constants (softmax)
            self.const_logits = nn.Parameter(torch.zeros(2))
        self.register_buffer("pop_seg", torch.zeros(()))
        self.register_buffer("pop_clf", torch.zeros(()))
        self.register_buffer("calibrated", torch.zeros(()))
        self.collect = None                                   # list while calibrating

    def weights(self, D_enh, f_enh):
        B = D_enh.shape[0]
        if self.mode == "off":
            return D_enh.new_ones(B, 2, dtype=torch.float32)
        if self.mode == "fixed":
            return D_enh.new_full((B, 2), 0.5, dtype=torch.float32)
        if self.mode == "constant":
            return torch.softmax(self.const_logits, 0).expand(B, 2)
        vs, vc = activation_dispersion(D_enh), activation_dispersion(f_enh)
        if self.collect is not None:
            self.collect.append((vs.sum().item(), vc.sum().item(), B))
        if not self.training and self.calibrated > 0:
            ms, mc = self.pop_seg, self.pop_clf                                         # inference: fixed means
        else:
            ms, mc = vs.mean(), vc.mean()                                               # training: batch mean
        z = torch.cat([vs / (ms + 1e-8), vc / (mc + 1e-8)], 1)                           # Eq. 5
        out = self.net(z)
        return torch.softmax(out, 1) if self.mode == "adaptive" else torch.sigmoid(out)  # Eq. 6

    def forward(self, D, D_enh, f, f_enh):
        with torch.autocast("cuda", enabled=False):
            D, D_enh, f, f_enh = D.float(), D_enh.float(), f.float(), f_enh.float()
            w = self.weights(D_enh, f_enh)
            D_fin = D + w[:, 0, None, None, None] * (D_enh - D)                          # Eq. 7
            f_fin = f + w[:, 1:2] * (f_enh - f)                                          # Eq. 8
        return D_fin, f_fin, w


class AdaptiveInteractionModel(nn.Module):
    """The proposed network. Encoder taps: expansion activations of the first blocks of stages 2, 3, 4 and 6, and the final activation (bridge)."""
    TAPS = [("s1", (1, 0)), ("s2", (2, 0)), ("s3", (3, 0)), ("s4", (5, 0))]

    def __init__(self, num_classes=3, mscf=True, tim=True, aiw="adaptive", direction="both",
                 dropout=0.3, pretrained=True):
        super().__init__()
        self.enc = timm.create_model("tf_efficientnet_b4.aa_in1k", pretrained=pretrained, drop_path_rate=0.2)
        for m in self.enc.modules():                         # batch-norm momentum 0.01 (running stats decay 0.99)
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01
        self.enc.classifier = nn.Identity()
        self._feats = {}
        for name, (i, j) in self.TAPS:
            self.enc.blocks[i][j].bn1.register_forward_hook(self._hook(name))
        self.enc.bn2.register_forward_hook(self._hook("br"))
        # frozen: stem, blocks 1-3 and block 4 up to its depthwise convolution
        b2b = self.enc.blocks[1][1]
        self.frozen = [self.enc.conv_stem, self.enc.bn1, self.enc.blocks[0], self.enc.blocks[1][0],
                       b2b.conv_pw, b2b.bn1, b2b.conv_dw, b2b.bn2]
        for m in self.frozen:
            for p in m.parameters():
                p.requires_grad = False
        ch = {"s1": 144, "s2": 192, "s3": 336, "s4": 960, "br": 1792}
        self.res = nn.ModuleDict({k: ResidualBlock(c, c, attention=True, mscf=mscf) for k, c in ch.items()})
        dec = [384, 192, 96, 48]
        skips, ups_in, clf_src = ["s4", "s3", "s2", "s1"], [1792, 384, 192, 96], ["br", "s4", "s3", "s2"]
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ci, c, 3, 2, padding=1, output_padding=1) for ci, c in zip(ups_in, dec)])
        self.gates = nn.ModuleList([AttentionGate(ch[s], c, c // 2) for s, c in zip(skips, dec)])
        self.dblocks = nn.ModuleList([ResidualBlock(c + ch[s], c, attention=False, mscf=mscf) for s, c in zip(skips, dec)])
        self.drop = nn.Dropout(dropout)
        self.clf_fc = nn.ModuleList([nn.Linear(ch[s], 256) for s in clf_src])
        self.tims = nn.ModuleList([TaskInteractionModule(c, 256, direction) for c in dec]) if tim else None
        self.aiws = nn.ModuleList([AdaptiveInteractionWeighting(aiw) for _ in dec]) if tim else None
        self.skips, self.clf_src = skips, clf_src
        self.seg_head = nn.Sequential(nn.ConvTranspose2d(48, 24, 3, 2, padding=1, output_padding=1),
                                      nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.clf_head = nn.Sequential(nn.Linear(192 + 336 + 960 + 1792 + 4 * 256, 256), nn.ReLU(),
                                      bn_layer(256, 1), nn.Dropout(0.3), nn.Linear(256, num_classes))
        for name, m in self.named_modules():                  # Glorot-uniform initialisation, zero biases, outside the encoder
            if not name.startswith("enc") and isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        mean, std = self.enc.pretrained_cfg["mean"], self.enc.pretrained_cfg["std"]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)

    def _hook(self, name):
        def f(module, inp, out):
            self._feats[name] = out
        return f

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen:                                 # frozen layers run in inference mode
            m.eval()
        return self

    def forward(self, x):                                     # x: [B, 3, 224, 224] in [0, 255]
        self._feats = {}
        self.enc.forward_features((x - self.mean) / self.std)
        e = {k: self.res[k](v) for k, v in self._feats.items()}
        d, clf_vecs, ws = e["br"], [], []
        for lvl in range(4):
            up = self.ups[lvl](d)
            skip = e[self.skips[lvl]]
            d = self.dblocks[lvl](torch.cat([up, self.gates[lvl](skip, up)], 1))
            f = F.relu(self.clf_fc[lvl](e[self.clf_src[lvl]].mean((2, 3))))
            if self.tims is not None:
                D_enh, f_enh = self.tims[lvl](d, f)
                d, f, w = self.aiws[lvl](d, D_enh, f, f_enh)
                ws.append(w)
            d = self.drop(d)          # after TIM/AIW: AIW sees the same (un-dropped) features in training and at test
            clf_vecs.append(f)
        with torch.autocast("cuda", enabled=False):
            seg = torch.sigmoid(self.seg_head(d.float()))
            z = torch.cat([e[k].float().mean((2, 3)) for k in ("s2", "s3", "s4", "br")] + [v.float() for v in clf_vecs], 1)
            clf = torch.softmax(self.clf_head(z), 1)
        return {"seg": seg, "clf": clf, "w": ws}


@torch.no_grad()
def calibrate_aiw(model, X, batch_size=24):
    """Fix each AIW level's inference means to the mean descriptor over X (un-augmented training
    images) in inference mode, level by level (d1 to d4), each with the upstream levels calibrated."""
    if model.aiws is None:
        return {}
    model.eval()
    stats = {}
    for lvl, a in enumerate(model.aiws):
        if a.mode not in ("adaptive", "sigmoid"):
            continue
        a.collect = []
        predict(model, X, batch_size, need=False)
        ss, sc, n = map(sum, zip(*a.collect))
        a.collect = None
        a.pop_seg.fill_(ss / n); a.pop_clf.fill_(sc / n); a.calibrated.fill_(1.0)
        stats[f"aiw_d{lvl + 1}"] = (ss / n, sc / n)
    return stats


def to_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()


@torch.no_grad()
def predict(model, X, batch_size, need=True):
    model.eval()
    segs, clfs, ws = [], [], []
    for i in range(0, len(X), batch_size):
        with torch.autocast("cuda", dtype=torch.float16, enabled=config.MIXED_PRECISION):
            o = model(to_tensor(X[i:i + batch_size]).to(DEVICE, non_blocking=True))
        if need:
            segs.append(o["seg"][:, 0].float().cpu()); clfs.append(o["clf"].float().cpu())
            ws.append([w.float().cpu() for w in o["w"]])
    if not need:
        return None
    w = [torch.cat([b[l] for b in ws]).numpy() for l in range(len(ws[0]))] if ws and ws[0] else []
    return torch.cat(segs).numpy(), torch.cat(clfs).numpy(), w


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


def total_loss(out, mask, label):
    return config.SEG_WEIGHT * seg_loss(mask, out["seg"].float()) + config.CLF_WEIGHT * clf_loss(label, out["clf"].float())


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
