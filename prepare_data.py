"""Link the downloaded BUSI and BUSI-WHU v3 archives into ./data in the layout
the loader expects (data/BUSI/<class>/<name>.png with <name>_mask.png, and
data/BUSI-WHU/<class>/<stem>.png with <stem>_mask.png).

BUSI-WHU v3 is read in the layout of the official archive (img/<id>.bmp,
gt/<id>_anno.bmp, class from the column "Benign(0) malignant(1)" of
Patient_infos_BUSI-WHU.xlsx) or, failing that, from benign/ and malignant/
folders. Images are paired with their masks by name; the script stops if any
image lacks its mask or a mask is empty. Expected counts: BUSI 133/437/210,
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

    def link_pairs(pairs, cls):
        dst = os.path.join("data", "BUSI-WHU", cls)
        os.makedirs(dst, exist_ok=True)
        for stem, img, msk in tqdm(pairs, desc=f"BUSI-WHU {cls}"):
            link(img, os.path.join(dst, f"{stem}.png"))               # OpenCV reads by content, not extension
            link(msk, os.path.join(dst, f"{stem}_mask.png"))
        bad = [stem for stem, _, msk in pairs if not (cv2.imread(msk, cv2.IMREAD_GRAYSCALE) > 127).any()]
        assert not bad, f"BUSI-WHU {cls}: masks empty after thresholding at 127: {bad[:5]}"
        return len(pairs)

    # (a) layout of the official v3 archive: img/, gt/ (<id>_anno.bmp) and Patient_infos_BUSI-WHU.xlsx
    official = []
    for root in ROOTS:
        for dirpath, _, files in os.walk(root):
            sd = subdirs(dirpath)
            if "img" in sd and "gt" in sd:
                xl = glob.glob(os.path.join(dirpath, "Patient_infos*.xlsx")) + \
                     glob.glob(os.path.join(os.path.dirname(dirpath), "Patient_infos*.xlsx"))
                if xl:
                    official.append((dirpath, xl[0]))
    if official:
        import pandas as pd
        WHU_SRC, info_path = official[0]
        print("BUSI-WHU (img/, gt/, patient table) found at", WHU_SRC)
        info = pd.read_excel(info_path)
        lab_col = next(c for c in info.columns if str(c).lower().startswith("benign"))
        img_by = {os.path.splitext(f)[0]: os.path.join(sd_, f) for sd_ in [subdirs(WHU_SRC)["img"]]
                  for f in os.listdir(sd_) if f.lower().endswith(IMG_EXT)}
        gt_by = {key_of(f, True): os.path.join(sd_, f) for sd_ in [subdirs(WHU_SRC)["gt"]]
                 for f in os.listdir(sd_) if f.lower().endswith(IMG_EXT)}
        pairs = {"benign": [], "malignant": []}
        missing = []
        for _, r in info.iterrows():
            stem = os.path.splitext(str(r["img_names"]))[0]
            if stem.lower() not in {k.lower() for k in img_by} or stem.lower() not in gt_by:
                missing.append(stem); continue
            img = next(v for k, v in img_by.items() if k.lower() == stem.lower())
            pairs["malignant" if int(r[lab_col]) == 1 else "benign"].append((stem, img, gt_by[stem.lower()]))
        assert not missing, f"BUSI-WHU: no image or mask for {missing[:5]}"
        for cls in ["benign", "malignant"]:
            whu_counts[cls] = link_pairs(sorted(pairs[cls]), cls)
    else:
        # (b) one folder per class (benign/, malignant/), images paired with masks by name
        whu_hits = [h for h in find_class_root(("benign", "malignant"), forbidden=("normal",))]
        whu_hits = [h for h in whu_hits if os.path.abspath(h) != os.path.abspath(BUSI_SRC)]
        assert whu_hits, "BUSI-WHU not found: add BUSI-WHU v3 (img/, gt/ and Patient_infos_BUSI-WHU.xlsx) as input."
        WHU_SRC = whu_hits[0]
        print("BUSI-WHU found at", WHU_SRC)
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
                    (masks if m else imgs).setdefault(key_of(rel, m), []).append(full)
            dup_i = [k for k, v in imgs.items() if len(v) > 1]
            dup_m = [k for k, v in masks.items() if len(v) > 1]
            missing = sorted(set(imgs) - set(masks))
            assert not dup_i and not dup_m and not missing, (
                f"BUSI-WHU {cls}: could not pair every image with one mask "
                f"(duplicate images {dup_i[:5]}, duplicate masks {dup_m[:5]}, no mask for {missing[:5]}).")
            pairs = [(os.path.splitext(os.path.basename(imgs[k][0]))[0], imgs[k][0], masks[k][0]) for k in sorted(imgs)]
            whu_counts[cls] = link_pairs(pairs, cls)
    print("BUSI-WHU images per class:", whu_counts)
    assert sum(whu_counts.values()) == 927, f"BUSI-WHU should hold 927 images, found {whu_counts}"


print(f"Linked into ./data: BUSI {busi_counts} | BUSI-WHU {whu_counts or 'skipped'}")
