# Tile the 0.5-MPP BCSS regions of interest into fixed-size tiles on the official site-disjoint split.
# Paper: --patch-size 448 (stride 224) and --patch-size 256 (stride 128); val/test tiles never overlap.
import argparse
import csv
import glob
import json
import os
import re
import time
from collections import Counter

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 2_000_000_000
NUM_CLASSES, IGNORE = 5, 0
CLASS_NAMES = ["background", "tumor", "stroma", "inflammatory", "necrosis"]
OFFICIAL_TEST_INST = ["OL", "LL", "E2", "EW", "GM", "S3"]   # BCSS reproducibility hold-out institutions


def patient_id(f):
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", f, re.I)
    return m.group(1).upper() if m else f


def tss_code(f):
    m = re.search(r"TCGA-([A-Z0-9]{2})-", f, re.I)
    return m.group(1).upper() if m else "NA"


def remap_mask(m, policy):
    out = np.zeros_like(m, dtype=np.uint8)
    for c in (1, 2, 3, 4):
        out[m == c] = c
    if policy == "official_merge":
        out[(m == 19) | (m == 20)] = 1      # angioinvasion, DCIS -> tumour
        out[(m == 10) | (m == 11)] = 3      # plasma cells, other immune -> inflammatory
    return out


def sat_map(rgb):
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return (mx - mn) / (mx + 1e-6)


def grid_coords(dim, size, stride):
    if dim < size:
        return []
    xs = list(range(0, dim - size + 1, stride))
    if xs and xs[-1] != dim - size:
        xs.append(dim - size)        # include the flush-right / flush-bottom tile
    return xs


def split_fovs(files, mode, test_inst, val_inst, val_frac, seed):
    tss = [tss_code(f) for f in files]
    rng = np.random.RandomState(seed)
    if mode == "official_site":
        test_set = set(test_inst)
        nontest = sorted(set(tss) - test_set)
        if val_inst:
            val_set = set(val_inst)
        else:
            order = list(nontest)
            rng.shuffle(order)
            cnt = Counter(tss)
            n_nontest = sum(cnt[i] for i in nontest)
            val_set, acc = set(), 0
            for i in order:
                if acc >= val_frac * n_nontest:
                    break
                val_set.add(i)
                acc += cnt[i]
        assert not (val_set & test_set), "val institutions overlap test"
        return {f: ("test" if t in test_set else "val" if t in val_set else "train") for f, t in zip(files, tss)}
    pats = sorted(set(patient_id(f) for f in files))
    rng.shuffle(pats)
    ntr, nva = int(0.70 * len(pats)), int(0.15 * len(pats))
    ps = {p: ("train" if i < ntr else "val" if i < ntr + nva else "test") for i, p in enumerate(pats)}
    return {f: ps[patient_id(f)] for f in files}


def main(argv=None):
    p = argparse.ArgumentParser(description="Tile 0.5-MPP BCSS ROIs into fixed-size tiles on the official site-disjoint split.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", required=True, help="folder with images/ and masks/ (BCSS ROIs at 0.5 MPP, same file names)")
    p.add_argument("--out", required=True, help="output root -> <out>/{train,val,test}/{image,mask}")
    p.add_argument("--patch-size", type=int, default=448, help="tile size in px (paper: 448 or 256)")
    p.add_argument("--stride", type=int, default=None, help="train stride (default patch-size/2 = 50%% overlap)")
    p.add_argument("--eval-stride", type=int, default=None, help="val/test stride (default patch-size = no overlap)")
    p.add_argument("--min-valid-frac", type=float, default=0.10, help="min fraction of clinical (1-4) pixels")
    p.add_argument("--min-tissue-frac", type=float, default=0.40, help="min fraction of tissue (saturation > 0.05)")
    p.add_argument("--label-policy", choices=["strict", "official_merge"], default="strict")
    p.add_argument("--split-mode", choices=["official_site", "random_patient"], default="official_site")
    p.add_argument("--test-inst", nargs="*", default=OFFICIAL_TEST_INST)
    p.add_argument("--val-inst", nargs="*", default=None, help="fix the val institutions (default: drawn with --split-seed)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=0)
    a = p.parse_args(argv)
    stride = a.stride or a.patch_size // 2
    eval_stride = a.eval_stride or a.patch_size

    img_dir, msk_dir = os.path.join(a.src, "images"), os.path.join(a.src, "masks")
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not files:
        raise SystemExit("no PNG images in " + img_dir)
    split = split_fovs(files, a.split_mode, a.test_inst, a.val_inst, a.val_frac, a.split_seed)
    for s in ("train", "val", "test"):
        fs = [f for f in files if split[f] == s]
        print("  %-5s: %3d ROIs | %2d institutions | %3d patients" % (
            s, len(fs), len({tss_code(f) for f in fs}), len({patient_id(f) for f in fs})))
        for sub in ("image", "mask"):
            os.makedirs(os.path.join(a.out, s, sub), exist_ok=True)

    stats = {s: {"kept": 0, "discarded": 0} for s in ("train", "val", "test")}
    train_px = np.zeros(NUM_CLASSES, np.int64)
    manifest = []
    t0 = time.time()
    P = a.patch_size
    for k, f in enumerate(files):
        s = split[f]
        stem = os.path.splitext(os.path.basename(f))[0]
        rgb = np.array(Image.open(f).convert("RGB"))
        g = remap_mask(np.array(Image.open(os.path.join(msk_dir, os.path.basename(f)))), a.label_policy)
        if rgb.shape[:2] != g.shape:
            g = np.array(Image.fromarray(g).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
        sat = sat_map(rgb)
        h, w = g.shape
        st = eval_stride if s in ("val", "test") else stride
        for y in grid_coords(h, P, st):
            for x in grid_coords(w, P, st):
                pm = g[y:y + P, x:x + P]
                vf = float((pm > 0).mean())
                tf_ = float((sat[y:y + P, x:x + P] > 0.05).mean())
                if vf >= a.min_valid_frac and tf_ >= a.min_tissue_frac:
                    name = "%s_y%d_x%d.png" % (stem, y, x)
                    Image.fromarray(rgb[y:y + P, x:x + P]).save(os.path.join(a.out, s, "image", name))
                    Image.fromarray(pm).save(os.path.join(a.out, s, "mask", name))
                    stats[s]["kept"] += 1
                    if s == "train":
                        train_px += np.bincount(pm.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
                    manifest.append([s, name, os.path.basename(f), x, y, round(vf, 4), round(tf_, 4)])
                else:
                    stats[s]["discarded"] += 1
        if k % 25 == 0:
            print("  ROI %d/%d  kept %s  %.1f min" % (k + 1, len(files), {s: v["kept"] for s, v in stats.items()}, (time.time() - t0) / 60))
    print("kept/discarded:", stats)

    clin = train_px.astype(np.float64)
    clin[IGNORE] = 0
    freq = clin / (clin.sum() + 1e-9)
    inv = np.zeros(NUM_CLASSES)
    nz = freq > 0
    inv[nz] = 1.0 / freq[nz]
    w = np.zeros(NUM_CLASSES)
    w[1:] = inv[1:] / (inv[1:].mean() + 1e-9)
    json.dump({"weights": [float(round(x, 4)) for x in w], "class_names": CLASS_NAMES,
               "train_pixel_freq": [float(round(x, 6)) for x in freq], "train_pixels": [int(x) for x in train_px],
               "patch_size": P, "stride": stride, "eval_stride": eval_stride, "label_policy": a.label_policy,
               "split_mode": a.split_mode, "test_inst": a.test_inst},
              open(os.path.join(a.out, "class_weights.json"), "w"), indent=2)
    with open(os.path.join(a.out, "fov_split.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "patient", "tss", "split"])
        for f in files:
            wr.writerow([os.path.basename(f), patient_id(f), tss_code(f), split[f]])
    with open(os.path.join(a.out, "patch_manifest.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["split", "patch", "source_fov", "x", "y", "valid_frac", "tissue_frac"])
        wr.writerows(manifest)
    print("class weights (bg=0):", [float(round(x, 4)) for x in w])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
