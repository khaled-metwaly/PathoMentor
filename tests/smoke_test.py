# End-to-end smoke test on synthetic data (CPU, a few minutes):
#   synthetic ROIs -> make_patches (448 and 256) -> dummy teacher cache -> train both arms x 2 seeds x 2 FOVs
#   (1 epoch, random-init backbone) -> compare -> evaluate (with boundary collar) -> predict.
# The dummy teacher has the exact on-disk format of build_teacher_cache.py; it is NOT a real UNI2-h teacher.
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def sh(*args):
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True, cwd=ROOT)


def synth_rois(src, n_per_inst=2):
    os.makedirs(os.path.join(src, "images"), exist_ok=True)
    os.makedirs(os.path.join(src, "masks"), exist_ok=True)
    rng = np.random.RandomState(0)
    insts = ["A1", "A2", "B6", "OL", "E2"]          # OL/E2 fall in the official test set
    k = 0
    for inst in insts:
        for j in range(n_per_inst):
            h, w = 520, 600
            m = np.full((h, w), 2, np.uint8)        # stroma background
            m[60:300, 80:360] = 1                   # tumour
            m[320:470, 100:300] = 3                 # inflammatory
            m[100:260, 400:560] = 4                 # necrosis
            m[480:, :] = 0                          # outside ROI
            palette = np.array([[240, 240, 240], [120, 40, 140], [230, 150, 190], [70, 30, 120], [200, 120, 160]], np.uint8)
            img = palette[m].astype(np.int16) + rng.randint(-25, 25, (h, w, 3))
            name = "TCGA-%s-%04d-DX1_xmin0_ymin0_MPP-0.5000.png" % (inst, 1000 + k)
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(src, "images", name))
            Image.fromarray(m).save(os.path.join(src, "masks", name))
            k += 1


def dummy_teacher(train_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(os.listdir(os.path.join(train_dir, "image")))
    rng = np.random.RandomState(0)
    logits = rng.randn(len(names), 32, 32, 5).astype(np.float32)
    soft = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    np.save(os.path.join(out_dir, "uni2h_native_softmaps_32x32x5.npy"), soft.astype(np.float16))
    json.dump(names, open(os.path.join(out_dir, "uni2h_native_index.json"), "w"))


def main():
    tmp = tempfile.mkdtemp(prefix="pathomentor_smoke_")
    try:
        src = os.path.join(tmp, "rois")
        synth_rois(src)
        runs = os.path.join(tmp, "runs")
        for fov in (448, 256):
            data = os.path.join(tmp, "tiles%d" % fov)
            sh("teacher/make_patches.py", "--src", src, "--out", data, "--patch-size", str(fov), "--val-inst", "B6")
            dummy_teacher(os.path.join(data, "train"), os.path.join(tmp, "teacher%d" % fov))
            for arm in ("baseline", "mentor"):
                for seed in (0, 1):
                    cmd = ["-m", "pathomentor.train", "--arm", arm, "--fov", str(fov), "--seed", str(seed),
                           "--data-root", data, "--out-dir", runs, "--epochs", "1", "--backbone-weights", "none",
                           "--phys-batch", "2", "--eff-batch", "4", "--max-train-tiles", "8"]
                    if arm == "mentor":
                        cmd += ["--teacher-cache", os.path.join(tmp, "teacher%d" % fov, "uni2h_native_softmaps_32x32x5.npy")]
                    sh(*cmd)
        sh("-m", "pathomentor.compare", "--runs", runs, "--out-dir", os.path.join(tmp, "compare"))
        w = os.path.join(runs, "mentor_fov256_seed0", "mentor_fov256_seed0.weights.h5")
        sh("-m", "pathomentor.evaluate", "--weights", w, "--fov", "256", "--test-dir", os.path.join(tmp, "tiles256", "test"),
           "--out-dir", os.path.join(tmp, "eval"), "--boundary-k", "0", "4")
        sh("-m", "pathomentor.predict", "--weights", w, "--fov", "256", "--images", os.path.join(tmp, "tiles256", "test", "image"),
           "--out-dir", os.path.join(tmp, "pred"), "--overlay")
        assert os.listdir(os.path.join(tmp, "pred", "overlay"))
        assert os.path.isfile(os.path.join(tmp, "compare", "compare_stats.json"))
        print("\nSMOKE TEST PASSED")
    finally:
        if os.environ.get("KEEP_SMOKE") != "1":
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print("kept", tmp)


if __name__ == "__main__":
    main()
