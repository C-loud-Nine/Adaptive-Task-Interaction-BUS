"""
models.py - comparison methods for Table 1, re-implemented in PyTorch for a
224 x 224 x 3 input under one training protocol.

Every model returns a dict:
  "seg": lesion probability map [B, 1, 224, 224] (sigmoid)
  "clf": class probabilities [B, C] (softmax), absent for segmentation-only models
  "aux": optional list of ({"seg": ..., "clf": ...}, weight) deep-supervision
         outputs, used only by methods whose original training defines them
         (MTANet, MTI-Net: weight 1; MTL-OCA soft object regions: weight 0.4).

Group                       Methods            Notes
CNN (+ classification head) unet, att_unet,    trained from scratch as in the
                            unet_pp            original papers; head: GAP of the
                                               bottleneck -> FC 256 -> FC C
Transformer (seg only)      transunet          R50-ViT-B/16, ImageNet-21k (timm)
                            swin_unet          Swin-T encoder, ImageNet-1k (timm),
                                               symmetric Swin decoder
                            missformer         from scratch, as published
Multi-task                  mtan               SegNet-style shared network with
                                               task-specific attention modules
                            mtanet             ResNet-50 encoder (ImageNet),
                                               reverse addition attention decoder,
                                               classification from the encoder
                            mti_net            HRNet-W18 (ImageNet), multi-scale
                                               initial predictions, multi-modal
                                               distillation, feature propagation
                            mtl_oca            Res-UNet with object contextual
                                               attention on the segmentation
                                               branch, MLP classifier on backbone
                                               features
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
INCEPTION = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))       # ViT-R50 orig_in21k weights
# "norm": input normalisation of the pretrained backbone (images in [0, 1]); None for scratch models
MODEL_INFO = {
    "unet": {"seg_only": False, "pretrained": False, "norm": None},
    "att_unet": {"seg_only": False, "pretrained": False, "norm": None},
    "unet_pp": {"seg_only": False, "pretrained": False, "norm": None},
    "transunet": {"seg_only": True, "pretrained": True, "norm": INCEPTION},
    "swin_unet": {"seg_only": True, "pretrained": True, "norm": IMAGENET},
    "missformer": {"seg_only": True, "pretrained": False, "norm": None},
    "mtan": {"seg_only": False, "pretrained": False, "norm": None},
    "mtanet": {"seg_only": False, "pretrained": True, "norm": IMAGENET},
    "mti_net": {"seg_only": False, "pretrained": True, "norm": IMAGENET},
    "mtl_oca": {"seg_only": False, "pretrained": False, "norm": None},
}


def conv_bn_relu(cin, cout, k=3, s=1, p=None, d=1):
    p = (k // 2) * d if p is None else p
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))


class ClfHead(nn.Module):
    """Classification head added to the segmentation CNNs: GAP -> FC 256 -> FC C."""
    def __init__(self, cin, n):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, 256),
                                 nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.net(x)


def out(seg_logit, clf_logit=None, aux=None, size=224):
    seg_logit = seg_logit.float()            # output activations in float32 (as for the proposed model)
    if seg_logit.shape[-1] != size:
        seg_logit = F.interpolate(seg_logit, size=(size, size), mode="bilinear", align_corners=False)
    d = {"seg": torch.sigmoid(seg_logit)}
    if clf_logit is not None:
        d["clf"] = torch.softmax(clf_logit.float(), dim=1)
    if aux:
        d["aux"] = aux
    return d


# ===========================================================================
# U-Net, Attention U-Net, UNet++
# ===========================================================================
class UNet(nn.Module):
    """Ronneberger et al., MICCAI 2015 (with batch normalisation)."""
    def __init__(self, n_cls, ch=(64, 128, 256, 512, 1024), attention=False):
        super().__init__()
        self.enc = nn.ModuleList([DoubleConv(3, ch[0])] +
                                 [DoubleConv(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(2 * ch[i], ch[i]) for i in reversed(range(4))])
        self.att = nn.ModuleList([AttGate(ch[i], ch[i], ch[i] // 2) for i in reversed(range(4))]) \
            if attention else None
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.clf = ClfHead(ch[4], n_cls)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        bottleneck = x
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = skips[3 - j]
            if self.att is not None:
                s = self.att[j](s, x)
            x = dec(torch.cat([s, x], 1))
        return out(self.head(x), self.clf(bottleneck))


class AttGate(nn.Module):
    """Additive attention gate, Oktay et al., 2018."""
    def __init__(self, c_skip, c_gate, c_int):
        super().__init__()
        self.wx = nn.Sequential(nn.Conv2d(c_skip, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.wg = nn.Sequential(nn.Conv2d(c_gate, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.psi = nn.Sequential(nn.Conv2d(c_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x, g):
        return x * self.psi(F.relu(self.wx(x) + self.wg(g)))


class UNetPP(nn.Module):
    """UNet++ (Zhou et al., DLMIA 2018), nested dense skips, no deep supervision."""
    def __init__(self, n_cls, nb=(32, 64, 128, 256, 512)):
        super().__init__()
        self.c = nn.ModuleDict()
        for i in range(5):
            self.c[f"{i}_0"] = DoubleConv(3 if i == 0 else nb[i - 1], nb[i])
        for j in range(1, 5):
            for i in range(5 - j):
                self.c[f"{i}_{j}"] = DoubleConv(nb[i] * j + nb[i + 1], nb[i])
        self.head = nn.Conv2d(nb[0], 1, 1)
        self.clf = ClfHead(nb[4], n_cls)

    def forward(self, x):
        X = {}
        for i in range(5):
            X[(i, 0)] = self.c[f"{i}_0"](x if i == 0 else F.max_pool2d(X[(i - 1, 0)], 2))
        for j in range(1, 5):
            for i in range(5 - j):
                up = F.interpolate(X[(i + 1, j - 1)], scale_factor=2, mode="bilinear", align_corners=False)
                X[(i, j)] = self.c[f"{i}_{j}"](torch.cat([X[(i, k)] for k in range(j)] + [up], 1))
        return out(self.head(X[(0, 4)]), self.clf(X[(4, 0)]))


# ===========================================================================
# TransUNet (R50-ViT-B/16)
# ===========================================================================
class TransUNet(nn.Module):
    """Chen et al., 2021/2024: R50 (ResNetV2 stem + 3 stages) + 12-layer ViT-B/16
    on 14 x 14 tokens, cascaded upsampler with skips from the CNN at 1/2, 1/4, 1/8."""
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_base_r50_s16_224.orig_in21k", pretrained=pretrained,
                                     num_classes=0)
        bb = self.vit.patch_embed.backbone
        self.stem, self.stages = bb.stem, bb.stages
        self.conv_more = conv_bn_relu(768, 512)
        dec_in = [512, 256, 128, 64]
        skip_ch = [512, 256, 64, 0]
        dec_out = [256, 128, 64, 16]
        self.blocks = nn.ModuleList([nn.Sequential(conv_bn_relu(dec_in[i] + skip_ch[i], dec_out[i]),
                                                   conv_bn_relu(dec_out[i], dec_out[i]))
                                     for i in range(4)])
        self.head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        # CNN path with skips: stem conv (1/2, before pooling), stage 1 (1/4), stage 2 (1/8)
        s = self.stem.conv(x)
        s = self.stem.norm(s)
        skip2 = s                                   # 64 @ 112
        h = self.stem.pool(s)
        feats = []
        for st in self.stages:
            h = st(h)
            feats.append(h)
        skip4, skip8 = feats[0], feats[1]           # 256 @ 56, 512 @ 28
        tok = self.vit.patch_embed.proj(h).flatten(2).transpose(1, 2)   # B, 196, 768
        tok = self.vit._pos_embed(tok)
        tok = self.vit.norm_pre(tok)
        tok = self.vit.blocks(tok)
        tok = self.vit.norm(tok)
        n_pref = getattr(self.vit, "num_prefix_tokens", 1)
        tok = tok[:, n_pref:]
        B, N, C = tok.shape
        g = int(math.sqrt(N))
        y = self.conv_more(tok.transpose(1, 2).reshape(B, C, g, g))     # 512 @ 14
        for blk, sk in zip(self.blocks, [skip8, skip4, skip2, None]):
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            if sk is not None:
                y = torch.cat([y, sk], 1)
            y = blk(y)
        return out(self.head(y))


# ===========================================================================
# Swin-UNet
# ===========================================================================
def window_partition(x, w):
    B, H, W, C = x.shape
    x = x.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, w * w, C)


def window_reverse(win, w, H, W):
    B = win.shape[0] // ((H // w) * (W // w))
    x = win.view(B, H // w, W // w, w, w, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, w, heads):
        super().__init__()
        self.w, self.heads, self.scale = w, heads, (dim // heads) ** -0.5
        self.rpb = nn.Parameter(torch.zeros((2 * w - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(w), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (w - 1)
        self.register_buffer("rpi", rel[..., 0] * (2 * w - 1) + rel[..., 1], persistent=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        a = q @ k.transpose(-2, -1) + self.rpb[self.rpi.view(-1)].view(N, N, -1).permute(2, 0, 1)[None]
        if mask is not None:
            nW = mask.shape[0]
            a = a.view(Bw // nW, nW, self.heads, N, N) + mask[None, :, None]
            a = a.view(Bw, self.heads, N, N)
        x = (a.softmax(-1) @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    def __init__(self, dim, res, heads, w=7, shift=0):
        super().__init__()
        self.res, self.w, self.shift = res, min(w, res), (0 if res <= w else shift)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.w, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        if self.shift:
            img = torch.zeros(1, res, res, 1)
            cnt = 0
            for hs in (slice(0, -self.w), slice(-self.w, -self.shift), slice(-self.shift, None)):
                for ws in (slice(0, -self.w), slice(-self.w, -self.shift), slice(-self.shift, None)):
                    img[:, hs, ws, :] = cnt
                    cnt += 1
            mw = window_partition(img, self.w).squeeze(-1)
            m = mw[:, None, :] - mw[:, :, None]
            self.register_buffer("mask", m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0),
                                 persistent=False)
        else:
            self.mask = None

    def forward(self, x):                      # x: B, H, W, C
        B, H, W, C = x.shape
        h = self.n1(x)
        if self.shift:
            h = torch.roll(h, (-self.shift, -self.shift), (1, 2))
        h = window_reverse(self.attn(window_partition(h, self.w), self.mask), self.w, H, W)
        if self.shift:
            h = torch.roll(h, (self.shift, self.shift), (1, 2))
        x = x + h
        return x + self.mlp(self.n2(x))


class PatchExpand(nn.Module):
    def __init__(self, dim, scale=2, out_dim=None):
        super().__init__()
        self.scale = scale
        out_dim = out_dim or dim // 2
        self.lin = nn.Linear(dim, out_dim * scale * scale, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):                      # B, H, W, C
        B, H, W, _ = x.shape
        x = self.lin(x)
        s = self.scale
        x = x.view(B, H, W, s, s, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, H * s, W * s, -1)
        return self.norm(x)


class SwinUNet(nn.Module):
    """Cao et al., ECCV-W 2022: Swin-T encoder (ImageNet-1k) and a symmetric Swin
    decoder with patch expanding and skip connections, final x4 expanding."""
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        import timm
        self.enc = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained,
                                     features_only=True, out_indices=(0, 1, 2, 3))
        dims, res, heads = [96, 192, 384, 768], [56, 28, 14, 7], [3, 6, 12, 24]
        self.up = nn.ModuleList([PatchExpand(dims[i]) for i in (3, 2, 1)])
        self.cat = nn.ModuleList([nn.Linear(2 * dims[i], dims[i]) for i in (2, 1, 0)])
        self.dec = nn.ModuleList([nn.Sequential(SwinBlock(dims[i], res[i], heads[i]),
                                                SwinBlock(dims[i], res[i], heads[i], shift=3))
                                  for i in (2, 1, 0)])
        self.norm = nn.LayerNorm(dims[0])
        self.final = PatchExpand(dims[0], scale=4, out_dim=dims[0])
        self.head = nn.Conv2d(dims[0], 1, 1)

    def forward(self, x):
        f = self.enc(x)
        f = [t if t.shape[-1] in (96, 192, 384, 768) and t.shape[1] == t.shape[2] else t.permute(0, 2, 3, 1)
             for t in f]                      # NHWC
        y = f[3]
        for up, cat, dec, sk in zip(self.up, self.cat, self.dec, (f[2], f[1], f[0])):
            y = up(y)
            y = dec(cat(torch.cat([y, sk], -1)))
        y = self.final(self.norm(y))           # B, 224, 224, 96
        return out(self.head(y.permute(0, 3, 1, 2)))


# ===========================================================================
# MISSFormer
# ===========================================================================
class MFEffAttention(nn.Module):
    """EfficientSelfAtten: spatial-reduction attention with a strided conv (ratio r)."""
    def __init__(self, dim, heads, r):
        super().__init__()
        self.heads, self.scale, self.r = heads, (dim // heads) ** -0.5, r
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        if r > 1:
            self.sr = nn.Conv2d(dim, dim, r, r)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        if self.r > 1:
            x = self.norm(self.sr(x.transpose(1, 2).reshape(B, C, H, W)).reshape(B, C, -1).transpose(1, 2))
        kv = self.kv(x).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = (q @ kv[0].transpose(-2, -1)) * self.scale
        return self.proj((a.softmax(-1) @ kv[1]).transpose(1, 2).reshape(B, N, C))

class MFMixFFNSkip(nn.Module):
    """MixFFN_skip: fc1 -> GELU(LN(DWConv(fc1 x) + fc1 x)) -> fc2. The official module also
    creates norm2 and norm3, which its forward never uses (kept when keep_unused=True)."""
    def __init__(self, c1, c2, keep_unused=True):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        if keep_unused:
            self.norm2, self.norm3 = nn.LayerNorm(c2), nn.LayerNorm(c2)

    def forward(self, x, H, W):
        B, N, _ = x.shape
        h = self.fc1(x)
        d = self.dwconv(h.transpose(1, 2).reshape(B, -1, H, W)).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(self.norm1(d + h)))

class MFBlock(nn.Module):
    """TransformerBlock(dim, head, reduction_ratio, token_mlp='mix_skip')."""
    def __init__(self, dim, heads, r, keep_unused=True):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MFEffAttention(dim, heads, r)
        self.mlp = MFMixFFNSkip(dim, dim * 4, keep_unused)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        return x + self.mlp(self.norm2(x), H, W)

class MFPatchEmbed(nn.Module):
    def __init__(self, cin, dim, k, s, p):
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, k, s, p)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W

class MFBridgeLayer(nn.Module):
    """BridgeLayer_4. The four encoder outputs (NHWC-flattened) are reshaped into 64-d tokens
    (stage widths 64/128/320/512 -> 1x/2x/5x/8x tokens) and concatenated (6076 tokens at 224).
    Attention (M_EfficientSelfAtten, 1 head): queries from all tokens, keys/values from
    Scale_reduce (strided convs at native widths 64/128/320 with ratios 8/4/2, stage 4 kept),
    then LayerNorm and per-scale MixFFN_skip at the native widths (hidden 4x)."""
    def __init__(self, c=64, mult=(1, 2, 5, 8), sizes=(56, 28, 14, 7), ratios=(8, 4, 2), keep_unused=True):
        super().__init__()
        self.c, self.mult, self.sizes = c, mult, sizes
        self.lens = [s * s * m for s, m in zip(sizes, mult)]
        self.norm1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c)
        self.kv = nn.Linear(c, 2 * c)
        self.proj = nn.Linear(c, c)
        self.sr = nn.ModuleList([nn.Conv2d(c * m, c * m, r, r) for m, r in zip(mult[:3], ratios)])
        self.sr_norm = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.mixffn = nn.ModuleList([MFMixFFNSkip(c * m, c * m * 4, keep_unused) for m in mult])

    def attn(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, C).transpose(1, 2)
        parts = torch.split(x, self.lens, 1)
        red = []
        for i in range(3):
            m, s = self.mult[i], self.sizes[i]
            t = parts[i].reshape(B, s, s, C * m).permute(0, 3, 1, 2)
            red.append(self.sr[i](t).reshape(B, C, -1).permute(0, 2, 1))
        red.append(parts[3])
        kvx = self.sr_norm(torch.cat(red, 1))
        kv = self.kv(kvx).reshape(B, -1, 2, 1, C).permute(2, 0, 3, 1, 4)
        a = (q @ kv[0].transpose(-2, -1)) * C ** -0.5
        return self.proj((a.softmax(-1) @ kv[1]).transpose(1, 2).reshape(B, N, C))

    def forward(self, x):
        B, _, C = x.shape
        tx1 = x + self.attn(self.norm1(x))
        tx = self.norm2(tx1)
        outs = []
        for p, m, s, ffn in zip(torch.split(tx, self.lens, 1), self.mult, self.sizes, self.mixffn):
            outs.append(ffn(p.reshape(B, -1, C * m), s, s).reshape(B, -1, C))
        return tx1 + torch.cat(outs, 1)

class MFPatchExpand(nn.Module):
    """PatchExpand (x2, dim -> dim/2) and FinalPatchExpand_X4 (x4, dim -> dim)."""
    def __init__(self, dim, scale=2):
        super().__init__()
        self.s = scale
        self.expand = nn.Linear(dim, 2 * dim if scale == 2 else 16 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2 if scale == 2 else dim)

    def forward(self, x, H, W):
        B = x.shape[0]
        x = self.expand(x)
        s = self.s
        x = x.view(B, H, W, s, s, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, H * s * W * s, -1)
        return self.norm(x)

class MFDecoderLayer(nn.Module):
    """MyDecoderLayer: concat with the bridge skip, Linear, two transformer blocks, patch
    expanding. The bottleneck layer (decoder_3) only expands; the official module still
    creates its concat Linear and two 512-d blocks (kept when keep_unused=True)."""
    def __init__(self, in_out, heads, r, n_out=1, is_last=False, is_first=False, keep_unused=True):
        super().__init__()
        dims, out_dim = in_out
        self.is_first, self.is_last = is_first, is_last
        if not is_first or keep_unused:
            self.concat_linear = nn.Linear(dims * (4 if is_last else 2), out_dim)
            self.layer_former_1 = MFBlock(out_dim, heads, r, keep_unused)
            self.layer_former_2 = MFBlock(out_dim, heads, r, keep_unused)
        self.layer_up = MFPatchExpand(out_dim, 4 if is_last else 2)
        self.last_layer = nn.Conv2d(out_dim, n_out, 1) if is_last else None
        for m in self.modules():                       # official init_weights
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x1, H, W, x2=None):
        if x2 is None:                                 # decoder_3: expand only
            return self.layer_up(x1, H, W)
        y = self.concat_linear(torch.cat([x1, x2], -1))
        y = self.layer_former_2(self.layer_former_1(y, H, W), H, W)
        y = self.layer_up(y, H, W)
        if self.last_layer is not None:
            B = y.shape[0]
            y = self.last_layer(y.view(B, 4 * H, 4 * W, -1).permute(0, 3, 1, 2))
        return y

class MISSFormer(nn.Module):
    """Huang et al., IEEE TMI 2023, following the official code (ZhifangDeng/MISSFormer):
    MiT-style encoder (dims 64/128/320/512, 2 blocks per stage, heads 1/2/5/8, reduction
    8/4/2/1, MixFFN_skip), Enhanced Transformer Context Bridge (4 BridgeLayer_4: stage tokens
    reshaped into 64-d tokens, joint spatial-reduction attention, per-scale MixFFN_skip at
    native widths), and the official decoder (in_out_chan [[32,64],[144,128],[288,320],[512,512]]).
    keep_unused=True keeps the modules the official code creates but never calls (decoder_3
    blocks and concat Linear, MixFFN_skip norm2/norm3; 7.01 M) so the count equals the
    official 42.46 M; keep_unused=False gives the identical function with 35.45 M."""
    def __init__(self, n_cls, img_size=224, keep_unused=True):
        super().__init__()
        dims, heads, sr = (64, 128, 320, 512), (1, 2, 5, 8), (8, 4, 2, 1)
        self.embeds = nn.ModuleList([MFPatchEmbed(3, dims[0], 7, 4, 3)] +
                                    [MFPatchEmbed(dims[i - 1], dims[i], 3, 2, 1) for i in range(1, 4)])
        self.enc = nn.ModuleList([nn.ModuleList([MFBlock(dims[i], heads[i], sr[i], keep_unused) for _ in range(2)])
                                  for i in range(4)])
        self.enorm = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        b = img_size // 4
        sizes = (b, b // 2, b // 4, b // 8)
        self.bridge = nn.ModuleList([MFBridgeLayer(64, (1, 2, 5, 8), sizes, keep_unused=keep_unused)
                                     for _ in range(4)])
        io = [[32, 64], [144, 128], [288, 320], [512, 512]]
        self.decoder_3 = MFDecoderLayer(io[3], heads[3], sr[3], is_first=True, keep_unused=keep_unused)
        self.decoder_2 = MFDecoderLayer(io[2], heads[2], sr[2], keep_unused=keep_unused)
        self.decoder_1 = MFDecoderLayer(io[1], heads[1], sr[1], keep_unused=keep_unused)
        self.decoder_0 = MFDecoderLayer(io[0], heads[0], sr[0], n_out=1, is_last=True, keep_unused=keep_unused)

    def forward(self, x):
        B = x.shape[0]
        feats, shapes, h = [], [], x
        for emb, blocks, nrm in zip(self.embeds, self.enc, self.enorm):
            t, H, W = emb(h)
            for blk in blocks:
                t = blk(t, H, W)
            t = nrm(t)
            feats.append(t); shapes.append((H, W))
            h = t.transpose(1, 2).reshape(B, -1, H, W)
        # bridge: NHWC-flattened stage tokens viewed as 64-d tokens
        z = torch.cat([f.reshape(B, -1, 64) for f in feats], 1)
        for layer in self.bridge:
            z = layer(z)
        parts = torch.split(z, self.bridge[0].lens, 1)
        sk = [p.reshape(B, -1, d) for p, d in zip(parts, (64, 128, 320, 512))]
        (H3, W3), (H2, W2), (H1, W1), (H0, W0) = shapes[3], shapes[2], shapes[1], shapes[0]
        y = self.decoder_3(sk[3], H3, W3)
        y = self.decoder_2(y, H2, W2, sk[2])
        y = self.decoder_1(y, H1, W1, sk[1])
        y = self.decoder_0(y, H0, W0, sk[0])          # B, 1, 224, 224
        return out(y)


# ===========================================================================
# MTAN
# ===========================================================================
class MTAN(nn.Module):
    """Liu et al., CVPR 2019, following the official SegNet-MTAN: shared VGG-16 SegNet
    (2/2/3/3/3 conv layers per encoder stage, mirrored decoder, max-pooling indices), a
    task-specific attention network per task (encoder_att per task; encoder_block_att and
    decoder_block_att shared across tasks, as in the official code). The segmentation task
    uses encoder and decoder attention and the official prediction layer (conv 3x3 + conv
    1x1); the classification task uses the encoder attention network, pooled into the
    classifier (GAP -> FC 256 -> FC C), since it has no dense decoder output."""
    def __init__(self, n_cls, f=(64, 128, 256, 512, 512)):
        super().__init__()

        def conv(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def att(c0, c1, c2):
            return nn.Sequential(nn.Conv2d(c0, c1, 1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
                                 nn.Conv2d(c1, c2, 1), nn.BatchNorm2d(c2), nn.Sigmoid())

        self.tasks = ("seg", "clf")
        self.encoder_block = nn.ModuleList([conv(3, f[0])] + [conv(f[i], f[i + 1]) for i in range(4)])
        self.decoder_block = nn.ModuleList([conv(f[0], f[0])] + [conv(f[i + 1], f[i]) for i in range(4)])
        self.conv_block_enc = nn.ModuleList([conv(f[0], f[0])])
        self.conv_block_dec = nn.ModuleList([conv(f[0], f[0])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(conv(f[i + 1], f[i + 1]))
                self.conv_block_dec.append(conv(f[i], f[i]))
            else:
                self.conv_block_enc.append(nn.Sequential(conv(f[i + 1], f[i + 1]), conv(f[i + 1], f[i + 1])))
                self.conv_block_dec.append(nn.Sequential(conv(f[i], f[i]), conv(f[i], f[i])))
        self.encoder_att = nn.ModuleDict({t: nn.ModuleList(
            [att(f[0], f[0], f[0])] + [att(2 * f[i + 1], f[i + 1], f[i + 1]) for i in range(4)]) for t in self.tasks})
        self.decoder_att = nn.ModuleList(                                   # segmentation task
            [att(2 * f[0], f[0], f[0])] + [att(f[i + 1] + f[i], f[i], f[i]) for i in range(4)])
        self.encoder_block_att = nn.ModuleList([conv(f[0], f[1])])
        self.decoder_block_att = nn.ModuleList([conv(f[0], f[0])])
        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(conv(f[i + 1], f[i + 2]))
                self.decoder_block_att.append(conv(f[i + 1], f[i]))
            else:
                self.encoder_block_att.append(conv(f[i + 1], f[i + 1]))
                self.decoder_block_att.append(conv(f[i + 1], f[i + 1]))
        self.seg_head = nn.Sequential(nn.Conv2d(f[0], f[0], 3, padding=1), nn.Conv2d(f[0], 1, 1))
        self.clf_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(f[4], 256),
                                      nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n_cls))
        for m in self.modules():                                            # official initialisation
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        g_enc, idx, up, g_dec = [], [], [], []
        h = x
        for i in range(5):                                                  # shared encoder
            a = self.encoder_block[i](h)
            b = self.conv_block_enc[i](a)
            g_enc.append((a, b))
            h, ix = F.max_pool2d(b, 2, 2, return_indices=True)
            idx.append(ix)
        d = h
        for j in range(5):                                                  # shared decoder
            u = F.max_unpool2d(d, idx[4 - j], 2, 2, output_size=g_enc[4 - j][1].shape[-2:])
            a = self.decoder_block[4 - j](u)
            d = self.conv_block_dec[4 - j](a)
            up.append(u); g_dec.append(d)
        att_enc = {}
        for t in self.tasks:                                                # task encoder attention
            prev = None
            for j in range(5):
                a, b = g_enc[j]
                m = self.encoder_att[t][j](a if j == 0 else torch.cat([a, prev], 1))
                prev = F.max_pool2d(self.encoder_block_att[j](m * b), 2, 2)
            att_enc[t] = prev
        y = att_enc["seg"]                                                  # segmentation decoder attention
        for j in range(5):
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
            y = self.decoder_block_att[4 - j](y)
            m = self.decoder_att[4 - j](torch.cat([up[j], y], 1))
            y = m * g_dec[j]
        return out(self.seg_head(y), self.clf_head(att_enc["clf"]))


# ===========================================================================
# MTANet (reverse addition attention)
# ===========================================================================
class RAA(nn.Module):
    """Reverse addition attention: refine a coarse map with features weighted by
    the reverse of the upsampled coarse prediction, added back to the prediction."""
    def __init__(self, cin, c=64):
        super().__init__()
        self.conv = nn.Sequential(conv_bn_relu(cin, c, 1, p=0), conv_bn_relu(c, c), conv_bn_relu(c, c),
                                  nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, feat, coarse):
        coarse = F.interpolate(coarse, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        rev = 1 - torch.sigmoid(coarse)
        return coarse + self.conv(feat * rev)


class MTANet(nn.Module):
    """Ling et al., IEEE TMI 2023/2024: ResNet-50 encoder (ImageNet), an
    aggregation decoder over the top three stages giving a coarse map, reverse
    addition attention refinement at stages 4, 3 and 2 with deep supervision,
    and a classification branch on attention-weighted encoder features."""
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        import timm
        self.enc = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, features_only=True,
                                     out_indices=(1, 2, 3, 4))          # 256, 512, 1024, 2048
        ch = [256, 512, 1024, 2048]
        self.red = nn.ModuleList([conv_bn_relu(c, 64, 1, p=0) for c in ch[1:]])
        self.agg = nn.Sequential(conv_bn_relu(192, 64), nn.Conv2d(64, 1, 1))
        self.raa = nn.ModuleList([RAA(ch[3]), RAA(ch[2]), RAA(ch[1])])
        self.att = nn.Sequential(nn.Conv2d(ch[3], 1, 1), nn.Sigmoid())
        self.clf = nn.Sequential(nn.Linear(ch[3], 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                 nn.Linear(256, n_cls))

    def forward(self, x):
        f1, f2, f3, f4 = self.enc(x)
        r = [self.red[0](f2), self.red[1](f3), self.red[2](f4)]
        size = r[0].shape[-2:]
        coarse = self.agg(torch.cat([F.interpolate(t, size=size, mode="bilinear", align_corners=False)
                                     for t in r], 1))
        m4 = self.raa[0](f4, coarse)
        m3 = self.raa[1](f3, m4)
        m2 = self.raa[2](f2, m3)
        a = self.att(f4)
        z = (f4 * a).sum((2, 3)) / a.sum((2, 3)).clamp(min=1e-6)
        clf = self.clf(z)
        aux = [(out(m), 1.0) for m in (coarse, m4, m3)]
        return out(m2, clf, aux)


# ===========================================================================
# MTI-Net
# ===========================================================================
class MTINet(nn.Module):
    """Vandenhende et al., ECCV 2020 on HRNet-W18 (ImageNet): initial task
    predictions at scales 1/4-1/32 (deep supervision), multi-modal distillation
    at each scale, feature propagation from coarse to fine, and final heads on
    the aggregated distilled features. The classification task uses the same
    task-feature machinery with global pooling."""
    def __init__(self, n_cls, pretrained=True, c=64):
        super().__init__()
        import timm
        self.enc = timm.create_model("hrnet_w18", pretrained=pretrained, features_only=True,
                                     feature_location="", out_indices=(1, 2, 3, 4))   # raw branches 18/36/72/144
        chs = self.enc.feature_info.channels()
        self.tasks = ("seg", "clf")
        self.task_feat = nn.ModuleList([nn.ModuleDict({t: nn.Sequential(conv_bn_relu(ci, c), conv_bn_relu(c, c))
                                                        for t in self.tasks}) for ci in chs])
        self.init_seg = nn.ModuleList([nn.Conv2d(c, 1, 1) for _ in chs])
        self.init_clf = nn.ModuleList([nn.Linear(c, n_cls) for _ in chs])
        self.prop = nn.ModuleList([nn.ModuleDict({t: conv_bn_relu(2 * c, c) for t in self.tasks}) for _ in chs[:-1]])
        self.distil = nn.ModuleList([nn.ModuleDict({
            f"{s}->{t}": nn.ModuleDict({"gate": nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.Sigmoid()),
                                        "msg": conv_bn_relu(c, c)})
            for t in self.tasks for s in self.tasks if s != t}) for _ in chs])
        self.final_seg = nn.Sequential(conv_bn_relu(c * len(chs), c), nn.Conv2d(c, 1, 1))
        self.final_clf = nn.Sequential(nn.Linear(c * len(chs), 256), nn.ReLU(inplace=True),
                                       nn.Dropout(0.3), nn.Linear(256, n_cls))

    def forward(self, x):
        feats = self.enc(x)                     # strides 4, 8, 16, 32
        n = len(feats)
        tf = [None] * n
        aux = []
        prev = None
        for i in reversed(range(n)):            # coarse to fine
            cur = {t: self.task_feat[i][t](feats[i]) for t in self.tasks}
            if prev is not None:                # feature propagation from the coarser scale
                cur = {t: self.prop[i][t](torch.cat([cur[t], F.interpolate(prev[t], size=cur[t].shape[-2:],
                                                                             mode="bilinear", align_corners=False)], 1))
                       for t in self.tasks}
            aux.append((out(self.init_seg[i](cur["seg"]), self.init_clf[i](cur["clf"].mean((2, 3)))), 1.0))
            dist = {}
            for t in self.tasks:
                d = cur[t]
                for s in self.tasks:
                    if s != t:
                        m = self.distil[i][f"{s}->{t}"]
                        d = d + m["gate"](cur[s]) * m["msg"](cur[s])
                dist[t] = d
            tf[i] = dist
            prev = cur
        size = tf[0]["seg"].shape[-2:]
        seg_cat = torch.cat([F.interpolate(tf[i]["seg"], size=size, mode="bilinear", align_corners=False)
                             for i in range(n)], 1)
        clf_cat = torch.cat([tf[i]["clf"].mean((2, 3)) for i in range(n)], 1)
        return out(self.final_seg(seg_cat), self.final_clf(clf_cat), aux)


# ===========================================================================
# MTL-OCA
# ===========================================================================
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout),
                                nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + self.skip(x))


class OCR(nn.Module):
    """Object contextual representation (Yuan et al., ECCV 2020) with K soft
    object regions, as used by MTL-OCA."""
    def __init__(self, c, k=2, key=128):
        super().__init__()
        self.soft = nn.Conv2d(c, k, 1)
        self.phi = nn.Sequential(conv_bn_relu(c, key, 1, p=0), conv_bn_relu(key, key, 1, p=0))
        self.psi = nn.Sequential(nn.Linear(c, key), nn.ReLU(inplace=True), nn.Linear(key, key))
        self.delta = nn.Linear(c, key)
        self.rho = conv_bn_relu(key, c, 1, p=0)
        self.g = conv_bn_relu(2 * c, c, 1, p=0)

    def forward(self, x):
        B, C, H, W = x.shape
        regions = self.soft(x)                                       # B, K, H, W (supervised)
        r = regions.flatten(2).softmax(-1)                           # spatial softmax
        f = r @ x.flatten(2).transpose(1, 2)                         # B, K, C object-region features
        q = self.phi(x).flatten(2).transpose(1, 2)                   # B, HW, key
        w = (q @ self.psi(f).transpose(1, 2) / math.sqrt(q.shape[-1])).softmax(-1)   # B, HW, K
        y = (w @ self.delta(f)).transpose(1, 2).reshape(B, -1, H, W)
        return self.g(torch.cat([x, self.rho(y)], 1)), regions


class MTLOCA(nn.Module):
    """Lu et al., Front. Oncol. 2025: three-level Res-UNet with GroupNorm, an
    object contextual attention module on the segmentation branch (K = 2 soft
    object regions, supervised as a coarse mask with weight 0.4), and a
    two-layer MLP classifier on the backbone features."""
    def __init__(self, n_cls, ch=(64, 128, 256, 512), sigma=1.0):
        super().__init__()
        # input: grayscale image and its Gaussian-derivative gradient magnitude (2 channels)
        r = int(3 * sigma)
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
        dg = -t / sigma ** 2 * g
        self.register_buffer("kx", (g[:, None] * dg[None, :])[None, None])
        self.register_buffer("ky", (dg[:, None] * g[None, :])[None, None])
        self.enc = nn.ModuleList([ResBlock(2, ch[0])] + [ResBlock(ch[i], ch[i + 1]) for i in range(3)])
        self.dec = nn.ModuleList([ResBlock(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(3))])
        self.ocr = OCR(ch[0])
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.clf = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[3], 256),
                                 nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n_cls))

    def forward(self, x):
        gray = x.mean(1, keepdim=True)
        pad = self.kx.shape[-1] // 2
        gp = F.pad(gray, (pad, pad, pad, pad), mode="reflect")
        mag = torch.sqrt(F.conv2d(gp, self.kx.to(gp.dtype)) ** 2 + F.conv2d(gp, self.ky.to(gp.dtype)) ** 2 + 1e-12)
        skips, h = [], torch.cat([gray, mag], 1)
        for i, e in enumerate(self.enc):
            h = e(h if i == 0 else F.max_pool2d(h, 2))
            skips.append(h)
        y = h
        for j, d in enumerate(self.dec):
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = d(torch.cat([y, skips[2 - j]], 1))
        z, regions = self.ocr(y)
        clf = self.clf(h)
        # soft object regions supervised as a coarse lesion map (channel 1 = lesion)
        coarse = regions[:, 1:2] - regions[:, 0:1]
        return out(self.head(z), clf, aux=[(out(coarse), 0.4)])


def build_model(name, num_classes, pretrained=None):
    """pretrained=None uses MODEL_INFO (ImageNet weights where the method uses them)."""
    pt = MODEL_INFO[name]["pretrained"] if pretrained is None else pretrained
    builders = {
        "unet": lambda: UNet(num_classes),
        "att_unet": lambda: UNet(num_classes, attention=True),
        "unet_pp": lambda: UNetPP(num_classes),
        "transunet": lambda: TransUNet(num_classes, pretrained=pt),
        "swin_unet": lambda: SwinUNet(num_classes, pretrained=pt),
        "missformer": lambda: MISSFormer(num_classes),
        "mtan": lambda: MTAN(num_classes),
        "mtanet": lambda: MTANet(num_classes, pretrained=pt),
        "mti_net": lambda: MTINet(num_classes, pretrained=pt),
        "mtl_oca": lambda: MTLOCA(num_classes),
    }
    if name not in builders:
        raise ValueError(name)
    return builders[name]()
