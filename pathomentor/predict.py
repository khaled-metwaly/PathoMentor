import argparse
import glob
import os

from .determinism import set_deterministic_env

set_deterministic_env()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from .data import make_decode_image_clean  # noqa: E402
from .model import build_deeplabv3plus  # noqa: E402

# bg, tumour, stroma, inflammatory, necrosis
PALETTE = np.array([[0, 0, 0], [214, 39, 40], [31, 119, 180], [44, 160, 44], [148, 103, 189]], np.uint8)


def overlay(img_u8, label, alpha=0.4):
    col = PALETTE[np.clip(label, 0, 4)]
    return ((1 - alpha) * img_u8 + alpha * col).astype(np.uint8)


def main(argv=None):
    p = argparse.ArgumentParser(description="Segment H&E tiles with a trained model (no teacher needed).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--fov", type=int, required=True, help="tile size the weights were trained at (448 / 256)")
    p.add_argument("--images", required=True, help="folder of PNG tiles at 0.5 MPP (or a single file)")
    p.add_argument("--out-dir", default="predictions")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--overlay", action="store_true", help="also save 40%% colour overlays")
    a = p.parse_args(argv)

    paths = [a.images] if os.path.isfile(a.images) else sorted(glob.glob(os.path.join(a.images, "*.png")))
    if not paths:
        raise SystemExit("no PNG files found in " + a.images)
    model = build_deeplabv3plus(image_size=a.fov, imagenet=False)
    model.load_weights(a.weights)
    dec = make_decode_image_clean(a.fov)
    os.makedirs(os.path.join(a.out_dir, "mask"), exist_ok=True)
    if a.overlay:
        os.makedirs(os.path.join(a.out_dir, "overlay"), exist_ok=True)
    for i in range(0, len(paths), a.batch_size):
        chunk = paths[i:i + a.batch_size]
        x = tf.stack([dec(q) for q in chunk])
        pred = tf.argmax(model(x, training=False), -1).numpy().astype(np.uint8)
        for q, m in zip(chunk, pred):
            name = os.path.basename(q)
            Image.fromarray(m).save(os.path.join(a.out_dir, "mask", name))   # label PNG: 1 tumour 2 stroma 3 inflam 4 necrosis
            if a.overlay:
                im = np.array(Image.open(q).convert("RGB").resize((a.fov, a.fov)))
                Image.fromarray(overlay(im, m)).save(os.path.join(a.out_dir, "overlay", name))
    print("segmented %d tiles -> %s" % (len(paths), a.out_dir))


if __name__ == "__main__":
    main()
