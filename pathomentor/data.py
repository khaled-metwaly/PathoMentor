import glob
import json
import os
import re

import numpy as np
import tensorflow as tf
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from . import NUM_CLASSES

# ---------------------------------------------------------------------------
# Augmentation (ported verbatim from the paper notebooks).
# Photometric ops never move pixels, so the cached teacher grid stays aligned.
# Geometric ops (flip_lr / rot90) are applied identically to image, mask and
# teacher grid.  No vertical flip is used.
# photometric codes: 0=identity 1=stain_jitter 2=hed_jitter 3=brightness_contrast 4=hue_sat
# geometric codes  : 0=none     1=flip_lr      2=rot90
# ---------------------------------------------------------------------------
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]], np.float32)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)
HED_SIGMA_A = 0.05
HED_SIGMA_B = 0.05


def _hed_perturb(img_u8, rs, channels):
    od = -np.log((img_u8.astype(np.float32) + 1.0) / 256.0)
    hed = od.reshape(-1, 3) @ _HED_FROM_RGB
    for c in channels:
        a = rs.uniform(1.0 - HED_SIGMA_A, 1.0 + HED_SIGMA_A)
        b = rs.uniform(-HED_SIGMA_B, HED_SIGMA_B)
        hed[:, c] = hed[:, c] * a + b
    od2 = hed @ _RGB_FROM_HED
    out = np.exp(-od2) * 256.0 - 1.0
    return np.clip(out, 0, 255).reshape(img_u8.shape).astype(np.uint8)


def photo_aug(img_u8, code, seed):
    code = int(code)
    if code == 0:
        return img_u8
    rs = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    if code == 1:
        return _hed_perturb(img_u8, rs, (0, 1))
    if code == 2:
        return _hed_perturb(img_u8, rs, (0, 1, 2))
    if code == 3:
        c = rs.uniform(0.8, 1.2)
        b = rs.uniform(-20.0, 20.0)
        out = (img_u8.astype(np.float32) - 128.0) * c + 128.0 + b
        return np.clip(out, 0, 255).astype(np.uint8)
    if code == 4:
        hsv = rgb_to_hsv(img_u8.astype(np.float32) / 255.0)
        hsv[..., 0] = (hsv[..., 0] + rs.uniform(-0.05, 0.05)) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * rs.uniform(0.8, 1.2), 0.0, 1.0)
        return np.clip(hsv_to_rgb(hsv) * 255.0, 0, 255).astype(np.uint8)
    return img_u8


def tf_photo(img_u8, code, seed):
    out = tf.numpy_function(photo_aug, [img_u8, code, seed], tf.uint8)
    out.set_shape([None, None, 3])
    return out


def geo_apply(geo, t):
    t = tf.cond(tf.equal(geo, 1), lambda: tf.image.flip_left_right(t), lambda: t)
    t = tf.cond(tf.equal(geo, 2), lambda: tf.image.rot90(t, 1), lambda: t)
    return t


def decode_u8(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=3)
    x.set_shape([None, None, 3])
    return x


def make_decode_mask(image_size):
    def decode_mask(path):
        m = tf.io.read_file(path)
        m = tf.image.decode_png(m, channels=1)
        m.set_shape([None, None, 1])
        m = tf.image.resize(m, [image_size, image_size], method="nearest")
        m = tf.cast(m, tf.int32)
        return tf.where(m > 4, tf.zeros_like(m), m)   # stray labels > 4 -> ignore (0)
    return decode_mask


def make_decode_image_clean(image_size):
    def decode_image_clean(path):
        x = decode_u8(path)
        x = tf.image.resize(tf.cast(x, tf.float32), [image_size, image_size])
        return tf.keras.applications.resnet50.preprocess_input(x)
    return decode_image_clean


# ---------------------------------------------------------------------------
# File indexing
# ---------------------------------------------------------------------------
def list_pairs(split_dir):
    imgs = sorted(glob.glob(os.path.join(split_dir, "image", "*.png")))
    masks = [os.path.join(split_dir, "mask", os.path.basename(p)) for p in imgs]
    missing = [m for m in masks if not os.path.isfile(m)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} masks missing in {split_dir}/mask (e.g. {missing[0]})")
    if not imgs:
        raise FileNotFoundError(f"no PNG tiles in {split_dir}/image")
    return imgs, masks


def patient_of(path, regex):
    m = re.search(regex, os.path.basename(path), re.I)
    return m.group(1).upper() if m else os.path.splitext(os.path.basename(path))[0]


class TeacherCache:
    # memory-mapped 32x32x5 soft class-maps, one row per TRAIN tile (by basename)
    def __init__(self, npy_path, index_path, teach_grid=32):
        self.arr = np.load(npy_path, mmap_mode="r")
        names = json.load(open(index_path))
        self.row = {b: i for i, b in enumerate(names)}
        if tuple(self.arr.shape[1:]) != (teach_grid, teach_grid, NUM_CLASSES):
            raise ValueError(f"teacher cache shape {self.arr.shape} != (N,{teach_grid},{teach_grid},{NUM_CLASSES})")

    def rows_for(self, img_paths):
        miss = [os.path.basename(p) for p in img_paths if os.path.basename(p) not in self.row]
        if miss:
            raise ValueError(f"{len(miss)} train tiles missing from teacher cache (e.g. {miss[0]}); rebuild the cache on this split")
        return np.array([self.row[os.path.basename(p)] for p in img_paths], np.int32)

    def fetch(self, r):
        return np.asarray(self.arr[int(r)], np.float32)


# ---------------------------------------------------------------------------
# tf.data builders
# ---------------------------------------------------------------------------
def _options():
    opt = tf.data.Options()
    opt.experimental_deterministic = True
    return opt


def build_train(cfg, seed, teacher=None):
    imgs, masks = list_pairs(cfg.train_dir)
    if cfg.max_train_tiles:
        imgs, masks = imgs[: cfg.max_train_tiles], masks[: cfg.max_train_tiles]
    n = len(imgs)
    size = cfg.fov
    decode_mask = make_decode_mask(size)
    augment = cfg.augment

    def draw():
        if augment:
            pc = tf.random.uniform([], 0, 5, tf.int32)
            sd = tf.random.uniform([], 0, 2147483647, tf.int64)
            gc = tf.random.uniform([], 0, 3, tf.int32)
        else:
            pc = tf.constant(0, tf.int32)
            sd = tf.constant(0, tf.int64)
            gc = tf.constant(0, tf.int32)
        return pc, sd, gc

    if teacher is not None:
        rows = teacher.rows_for(imgs)
        ds = tf.data.Dataset.from_tensor_slices((imgs, masks, rows)).shuffle(n, seed=seed, reshuffle_each_iteration=True)
        tg = cfg.teach_grid

        def _m(ip, mp, tr):
            pc, sd, gc = draw()
            img = tf_photo(decode_u8(ip), pc, sd)
            img = tf.image.resize(tf.cast(img, tf.float32), [size, size])
            msk = decode_mask(mp)
            tsoft = tf.numpy_function(teacher.fetch, [tr], tf.float32)
            tsoft.set_shape([tg, tg, NUM_CLASSES])
            img = geo_apply(gc, img)
            msk = geo_apply(gc, msk)
            tsoft = geo_apply(gc, tsoft)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            return img, (msk, tsoft)
    else:
        ds = tf.data.Dataset.from_tensor_slices((imgs, masks)).shuffle(n, seed=seed, reshuffle_each_iteration=True)

        def _m(ip, mp):
            pc, sd, gc = draw()
            img = tf_photo(decode_u8(ip), pc, sd)
            img = tf.image.resize(tf.cast(img, tf.float32), [size, size])
            msk = decode_mask(mp)
            img = geo_apply(gc, img)
            msk = geo_apply(gc, msk)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            return img, msk

    ds = ds.map(_m, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.phys_batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE).with_options(_options())
    return ds, n


def build_eval(split_dir, image_size, batch_size, with_paths=False):
    imgs, masks = list_pairs(split_dir)
    dimg = make_decode_image_clean(image_size)
    dmsk = make_decode_mask(image_size)
    ds = tf.data.Dataset.from_tensor_slices((imgs, masks)).map(lambda a, b: (dimg(a), dmsk(b)),
                                                                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).with_options(_options())
    return (ds, imgs) if with_paths else ds
