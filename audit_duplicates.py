"""
audit_duplicates.py - exact and near-duplicate audit for BUSI / BUSI-WHU.

Reads every image (masks excluded), groups exact duplicates by the SHA-256 of
the decoded pixels and near duplicates by a 64-bit difference hash (dHash)
within a Hamming distance, then checks each group against the split manifest
written by data_loader.prepare_datasets().

Usage:
    python audit_duplicates.py --root data/BUSI --manifest results/split_manifest_busi.csv
    python audit_duplicates.py --root data/BUSI-WHU --manifest results/split_manifest_busi_whu.csv

Writes results/duplicate_groups_<name>.csv and prints:
  * number of exact and near-duplicate groups,
  * groups whose members fall in different partitions (train/val/test),
  * groups whose members carry different class labels.
Near-duplicate groups need visual confirmation before any split is changed.
"""
import argparse, csv, hashlib, os, glob
from collections import defaultdict

import cv2
import numpy as np


def dhash(gray, size=8):
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hamming", type=int, default=4)
    args = ap.parse_args()

    split, label = {}, {}
    with open(args.manifest) as fh:
        for row in csv.DictReader(fh):
            split[row["file"]] = row["split"]
            label[row["file"]] = row["class"]

    exact, hashes = defaultdict(list), {}
    for rel in sorted(split):
        img = cv2.imread(os.path.join(args.root, rel))
        if img is None:
            continue
        exact[hashlib.sha256(img.tobytes()).hexdigest()].append(rel)
        hashes[rel] = dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # union-find over exact groups and near pairs
    parent = {f: f for f in hashes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for members in exact.values():
        for m in members[1:]:
            union(members[0], m)
    files = sorted(hashes)
    for i, a in enumerate(files):
        for b in files[i + 1:]:
            if bin(hashes[a] ^ hashes[b]).count("1") <= args.hamming:
                union(a, b)

    groups = defaultdict(list)
    for f in files:
        groups[find(f)].append(f)
    groups = [g for g in groups.values() if len(g) > 1]
    n_exact = sum(1 for g in exact.values() if len(g) > 1)
    cross = [g for g in groups if len({split[f] for f in g}) > 1]
    conflict = [g for g in groups if len({label[f] for f in g}) > 1]

    name = os.path.splitext(os.path.basename(args.manifest))[0].replace("split_manifest_", "")
    out = os.path.join(os.path.dirname(args.manifest), f"duplicate_groups_{name}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "file", "class", "split", "exact_sha256_group"])
        for gi, g in enumerate(groups):
            for f in g:
                sha = next(k for k, v in exact.items() if f in v)
                w.writerow([gi, f, label[f], split[f], len(exact[sha]) > 1])

    print(f"images: {len(files)}")
    print(f"exact-duplicate groups: {n_exact}")
    print(f"duplicate groups (exact + near, Hamming <= {args.hamming}): {len(groups)}")
    print(f"groups spanning partitions: {len(cross)}")
    print(f"groups with conflicting class labels: {len(conflict)}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
