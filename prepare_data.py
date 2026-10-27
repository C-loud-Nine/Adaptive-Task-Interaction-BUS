"""Link the downloaded BUSI and BUSI-WHU v3 archives into ./data in the layout
the loader expects (data/BUSI/<class>/<name>.png with <name>_mask.png, and
data/BUSI-WHU/<class>/<stem>.png with <stem>_mask.png).

BUSI-WHU images are paired with their masks by file name (mask files contain
"mask", "gt", "label", "annot" or "seg"); the script stops if any image lacks
exactly one mask or a mask is empty. Expected counts: BUSI 133/437/210,
BUSI-WHU 927.

    python prepare_data.py --input /path/to/downloads
"""
import argparse, os, re, glob
import cv2
from tqdm.auto import tqdm

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--input", nargs="+", default=["/kaggle/input"] if os.path.isdir("/kaggle/input") else ["input"],
                help="folder(s) that contain the downloaded BUSI and BUSI-WHU archives (searched recursively)")
ap.add_argument("--skip-whu", action="store_true", help="link BUSI only")
args = ap.parse_args()
ROOTS = [os.path.abspath(p) for p in args.input]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MASK_TOKENS = ("mask", "gt", "label", "annot", "anno", "seg")

def subdirs(p):
    try:
        return {d.lower(): os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))}
    except OSError:
        return {}

def find_class_root(required, forbidden=()):
    hits = []
    for root in ROOTS:
        for dirpath, dirnames, _ in os.walk(root):
            sd = subdirs(dirpath)
            if all(r in sd for r in required) and not any(f in sd for f in forbidden):
                hits.append(dirpath)
    return hits

def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)

# ---------------- BUSI: folders normal/benign/malignant with "<name>_mask*.png"
busi_hits = [h for h in find_class_root(("normal", "benign", "malignant"))
             if glob.glob(os.path.join(subdirs(h)["benign"], "*_mask*.png"))]
assert busi_hits, "BUSI not found: add the 'Breast Ultrasound Images Dataset' (folders normal/benign/malignant) as input."
BUSI_SRC = busi_hits[0]
print("BUSI found at", BUSI_SRC)
busi_counts = {}
for cls in ["normal", "benign", "malignant"]:
    src = subdirs(BUSI_SRC)[cls]
    dst = os.path.join("data", "BUSI", cls)
    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(".png"))
    for f in tqdm(files, desc=f"BUSI {cls}"):
        link(os.path.join(src, f), os.path.join(dst, f))
    busi_counts[cls] = sum(1 for f in files if "_mask" not in f)
print("BUSI images per class:", busi_counts)
assert busi_counts == {"normal": 133, "benign": 437, "malignant": 210}, (
    f"BUSI should hold 133/437/210 images, found {busi_counts}")

whu_counts = {}
if not args.skip_whu:
    # ---------------- BUSI-WHU v3: folders benign/malignant (no normal), images paired with masks
    def is_mask(rel):
        parts = re.split(r"[\\/_\-\s\.]+", rel.lower())
        return any(p.startswith(t) for p in parts for t in MASK_TOKENS)

    def key_of(rel, mask):
        stem = os.path.splitext(os.path.basename(rel))[0]
        if mask:
            stem = re.sub(r"[_\-\s]*(mask|gt|label|annotation|annot|anno|seg)[_\-\s]*\d*$", "", stem, flags=re.I)
            stem = re.sub(r"^(mask|gt|label|annotation|annot|anno|seg)[_\-\s]*", "", stem, flags=re.I)
        return stem.lower()

    whu_hits = [h for h in find_class_root(("benign", "malignant"), forbidden=("normal",))]
    whu_hits = [h for h in whu_hits if os.path.abspath(h) != os.path.abspath(BUSI_SRC)]
    assert whu_hits, "BUSI-WHU not found: add BUSI-WHU v3 (folders benign/malignant) as input."
    WHU_SRC = whu_hits[0]
    print("BUSI-WHU found at", WHU_SRC)
    whu_counts = {}
    for cls in ["benign", "malignant"]:
        src = subdirs(WHU_SRC)[cls]
        imgs, masks = {}, {}
        for dirpath, _, files in os.walk(src):
            for f in files:
                if not f.lower().endswith(IMG_EXT):
                    continue
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, src)
                m = is_mask(rel)
                k = key_of(rel, m)
                (masks if m else imgs).setdefault(k, []).append(full)
        dup_i = [k for k, v in imgs.items() if len(v) > 1]
        dup_m = [k for k, v in masks.items() if len(v) > 1]
        missing = sorted(set(imgs) - set(masks))
        assert not dup_i and not dup_m and not missing, (
            f"BUSI-WHU {cls}: could not pair every image with one mask "
            f"(duplicate images {dup_i[:5]}, duplicate masks {dup_m[:5]}, no mask for {missing[:5]}). "
            "Send this message and the folder layout.")
        dst = os.path.join("data", "BUSI-WHU", cls)
        os.makedirs(dst, exist_ok=True)
        for k in tqdm(sorted(imgs), desc=f"BUSI-WHU {cls}"):
            stem = os.path.splitext(os.path.basename(imgs[k][0]))[0]
            link(imgs[k][0], os.path.join(dst, f"{stem}.png"))       # OpenCV reads by content, not extension
            link(masks[k][0], os.path.join(dst, f"{stem}_mask.png"))
        bad = [k for k in sorted(imgs)
               if not (cv2.imread(masks[k][0], cv2.IMREAD_GRAYSCALE) > 127).any()]
        assert not bad, f"BUSI-WHU {cls}: masks empty after thresholding at 127: {bad[:5]}"
        whu_counts[cls] = len(imgs)
    print("BUSI-WHU images per class:", whu_counts)
    assert sum(whu_counts.values()) == 927, f"BUSI-WHU should hold 927 images, found {whu_counts}"


print(f"Linked into ./data: BUSI {busi_counts} | BUSI-WHU {whu_counts or 'skipped'}")
