# Build the offline PathoMentor teacher: UNI2-h at NATIVE resolution, dense, as 32x32x5 soft class-maps.
#   1. each train tile is resized to 448 px (a no-op for 448 tiles; an upscale for 256 tiles) and split into
#      the 2x2 grid of native 224 px windows -> 16x16 tokens each -> stitched 32x32x1536 token map
#   2. a small class-weighted MLP probe is trained on clinical tokens from a stratified subset of tiles
#   3. every TRAIN tile gets a 32x32x5 softmax map -> uni2h_native_softmaps_32x32x5.npy + uni2h_native_index.json
# The teacher is used only during training; it is never needed at inference.
# Requires: PyTorch, timm>=1.0.3, huggingface_hub, and access to the gated MahmoodLab/UNI2-h model (HF token).
import argparse
import glob
import json
import os
import re
import time

import numpy as np
from PIL import Image

TILE, WIN, NWIN, TOK = 448, 224, 2, 16
GRID = NWIN * TOK
UNI2_DIM, NUM_CLASSES, IGNORE = 1536, 5, 0


def tss_code(f):
    m = re.search(r"TCGA-([A-Z0-9]{2})-", f, re.I)
    return m.group(1).upper() if m else "NA"


def load_uni2h(device, hf_token=None):
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=UNI2_DIM,
              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    model = timm.create_model("hf_hub:MahmoodLab/UNI2-h", pretrained=True, **kw).to(device).eval()
    prep = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, prep, getattr(model, "num_prefix_tokens", 0)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the cached UNI2-h soft class-map teacher.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-dir", required=True, help="<dir>/image + <dir>/mask (the TRAIN split from make_patches.py)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (or env HF_TOKEN)")
    p.add_argument("--device", default=None, help="cuda / cpu (auto)")
    p.add_argument("--probe-tiles", type=int, default=1500)
    p.add_argument("--probe-tokens-per-tile", type=int, default=48)
    p.add_argument("--max-probe-samples", type=int, default=120000)
    p.add_argument("--probe-hidden", type=int, default=256)
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hours", type=float, default=11.0, help="stop and checkpoint (re-run to resume)")
    a = p.parse_args(argv)

    import torch
    import torch.nn as nn

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    deadline = time.time() + a.max_hours * 3600
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    cache_npy = os.path.join(a.out_dir, "uni2h_native_softmaps_32x32x5.npy")
    index_json = os.path.join(a.out_dir, "uni2h_native_index.json")
    done_json = os.path.join(a.out_dir, "uni2h_native_done.json")
    probe_pt = os.path.join(a.out_dir, "uni2h_probe.pt")

    imgs = sorted(glob.glob(os.path.join(a.train_dir, "image", "*.png")))
    if not imgs:
        raise SystemExit("no train tiles in " + a.train_dir)
    mask_for = lambda q: os.path.join(a.train_dir, "mask", os.path.basename(q))  # noqa: E731
    print("train tiles", len(imgs), "| device", dev)
    um, prep, npref = load_uni2h(dev, a.hf_token)

    @torch.inference_mode()
    def dense_tokens(path):
        im = Image.open(path).convert("RGB")
        if im.size != (TILE, TILE):
            im = im.resize((TILE, TILE))      # 256 tiles are upscaled to the 448-equivalent view
        wins = [prep(im.crop((wx * WIN, wy * WIN, (wx + 1) * WIN, (wy + 1) * WIN))) for wy in range(NWIN) for wx in range(NWIN)]
        f = um.forward_features(torch.stack(wins).to(dev))[:, npref:, :]
        f = f.reshape(NWIN * NWIN, TOK, TOK, UNI2_DIM).float().cpu().numpy()
        grid = np.zeros((GRID, GRID, UNI2_DIM), np.float32)
        i = 0
        for wy in range(NWIN):
            for wx in range(NWIN):
                grid[wy * TOK:(wy + 1) * TOK, wx * TOK:(wx + 1) * TOK] = f[i]
                i += 1
        return grid

    def mask_grid(path):
        return np.array(Image.open(path).resize((GRID, GRID), Image.NEAREST), np.int64)

    probe = nn.Sequential(nn.LayerNorm(UNI2_DIM), nn.Linear(UNI2_DIM, a.probe_hidden), nn.GELU(),
                          nn.Dropout(0.1), nn.Linear(a.probe_hidden, NUM_CLASSES)).to(dev)
    if os.path.exists(probe_pt):
        probe.load_state_dict(torch.load(probe_pt, map_location=dev))
        print("loaded trained probe", probe_pt)
    else:
        # PASS 1: stratified (by institution) probe-training tokens, clinical classes only
        rng = np.random.RandomState(a.seed)
        by_inst = {}
        for q in imgs:
            by_inst.setdefault(tss_code(q), []).append(q)
        per = max(1, a.probe_tiles // max(1, len(by_inst)))
        pool = []
        for lst in by_inst.values():
            lst = list(lst)
            rng.shuffle(lst)
            pool += lst[:per]
        rng.shuffle(pool)
        pool = pool[: a.probe_tiles]
        X, Y = [], []
        t0 = time.time()
        for i, q in enumerate(pool):
            g, lab = dense_tokens(q), mask_grid(mask_for(q))
            sel = np.argwhere(lab > 0)
            if len(sel) == 0:
                continue
            if len(sel) > a.probe_tokens_per_tile:
                sel = sel[rng.choice(len(sel), a.probe_tokens_per_tile, replace=False)]
            for r, c in sel:
                X.append(g[r, c].astype(np.float16))
                Y.append(int(lab[r, c]))
            if len(X) >= a.max_probe_samples:
                break
            if i % 200 == 0:
                print("  probe tokens: tile %d/%d, %d samples, %.1f min" % (i + 1, len(pool), len(X), (time.time() - t0) / 60))
        X = torch.tensor(np.stack(X).astype(np.float32), device=dev)
        Y = torch.tensor(np.array(Y, np.int64), device=dev)
        cnt = np.bincount(Y.cpu().numpy(), minlength=NUM_CLASSES).astype(np.float64)
        w = np.zeros(NUM_CLASSES)
        nz = cnt > 0
        w[nz] = 1.0 / cnt[nz]
        w[1:] = w[1:] / w[1:].mean()
        w[IGNORE] = 0.0
        opt = torch.optim.Adam(probe.parameters(), lr=a.probe_lr, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
        perm = torch.randperm(len(Y))
        cut = int(0.85 * len(Y))
        tr, va = perm[:cut], perm[cut:]
        probe.train()
        for _ in range(a.probe_epochs):
            pp = tr[torch.randperm(len(tr))]
            for j in range(0, len(pp), 4096):
                b = pp[j:j + 4096]
                opt.zero_grad()
                lossf(probe(X[b]), Y[b]).backward()
                opt.step()
        probe.eval()
        with torch.inference_mode():
            pv = probe(X[va]).argmax(1)
            bacc = [float((pv[Y[va] == c] == c).float().mean()) for c in (1, 2, 3, 4) if (Y[va] == c).sum() > 0]
        print("dense probe: token acc %.3f | balanced (clinical) %.3f | per class %s" % (
            float((pv == Y[va]).float().mean()), float(np.mean(bacc)), [round(x, 3) for x in bacc]))
        torch.save(probe.state_dict(), probe_pt)

    # PASS 2: soft class-maps for every TRAIN tile (resumable)
    base = [os.path.basename(q) for q in imgs]
    if os.path.exists(cache_npy) and os.path.exists(done_json):
        cache = np.load(cache_npy)
        done = set(json.load(open(done_json)))
        print("resuming: %d done" % len(done))
    else:
        cache = np.zeros((len(imgs), GRID, GRID, NUM_CLASSES), np.float16)
        done = set()
    idx = {b: i for i, b in enumerate(base)}
    t0, n = time.time(), 0
    for q in imgs:
        b = os.path.basename(q)
        if b in done:
            continue
        if time.time() >= deadline:
            print("[TimeGuard] stopping; re-run the same command to resume")
            break
        g = dense_tokens(q)
        with torch.inference_mode():
            soft = torch.softmax(probe(torch.tensor(g.reshape(-1, UNI2_DIM), device=dev)), -1)
        cache[idx[b]] = soft.reshape(GRID, GRID, NUM_CLASSES).float().cpu().numpy().astype(np.float16)
        done.add(b)
        n += 1
        if n % 500 == 0:
            np.save(cache_npy, cache)
            json.dump(sorted(done), open(done_json, "w"))
            print("  %d/%d tiles, %.1f min" % (len(done), len(imgs), (time.time() - t0) / 60))
    np.save(cache_npy, cache)
    json.dump(base, open(index_json, "w"))
    json.dump(sorted(done), open(done_json, "w"))
    json.dump(dict(grid=GRID, dim=UNI2_DIM, num_classes=NUM_CLASSES, tile=TILE, win=WIN, probe_hidden=a.probe_hidden,
                   n_train=len(imgs), n_done=len(done), cache=os.path.basename(cache_npy), index=os.path.basename(index_json)),
              open(os.path.join(a.out_dir, "probe_meta.json"), "w"), indent=2)
    print("soft-maps: %d/%d ->" % (len(done), len(imgs)), cache_npy)
    if len(done) < len(imgs):
        print("[note] cache incomplete; re-run to finish before training the mentor arm")


if __name__ == "__main__":
    main()
