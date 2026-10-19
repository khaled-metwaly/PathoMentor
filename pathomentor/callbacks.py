import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import CLASS_SHORT, NUM_CLASSES
from .metrics import all_class_metrics, confusion


def predict_logits(model, imgs):
    out = model(imgs, training=False)
    return out[0] if isinstance(out, (list, tuple)) else out


class SegMetrics(keras.callbacks.Callback):
    # pooled validation metrics each epoch; writes val_fwiou etc. into logs so that
    # EarlyStopping / ReduceLROnPlateau / restore_best_weights all select on them
    KEYS = ["val_clinical_miou", "val_clin3", "val_fwiou", "val_dice", "val_recall"]

    def __init__(self, val_ds, eval_model=None):
        super().__init__()
        self.val = val_ds
        self.eval_model = eval_model
        self.hist = {k: [] for k in self.KEYS}

    def on_epoch_end(self, epoch, logs=None):
        net = self.eval_model or self.model
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        for imgs, masks in self.val:
            pr = tf.argmax(predict_logits(net, imgs), -1).numpy()
            tr = tf.cast(tf.squeeze(masks, -1), tf.int32).numpy()
            cm += confusion(tr, pr)
        iou, dice, rec, _, freq = all_class_metrics(cm)
        clin = float(np.mean(iou[1:]))
        clin3 = float(np.mean(iou[1:4]))
        fw = float(np.sum(freq * np.array(iou)))
        d = float(np.mean(dice[1:]))
        r = float(np.mean(rec[1:]))
        for k, v in zip(self.KEYS, [clin, clin3, fw, d, r]):
            self.hist[k].append(v)
        if logs is not None:
            logs.update(dict(zip(self.KEYS, [clin, clin3, fw, d, r])))
        print("  val ClinMIoU4=%.2f ClinMIoU3=%.2f FWIoU=%.2f | " % (clin * 100, clin3 * 100, fw * 100)
              + " ".join("%s=%.1f" % (CLASS_SHORT[c], iou[c] * 100) for c in range(NUM_CLASSES)))


class AlphaSchedule(keras.callbacks.Callback):
    # linear warm-up 0 -> alpha_peak over `ramp` epochs, then held (no release during training)
    def __init__(self, alpha_peak=0.5, warmup=0, ramp=2):
        super().__init__()
        self.peak, self.warmup, self.ramp = alpha_peak, warmup, ramp

    def value(self, e):
        if e < self.warmup:
            return 0.0
        if e < self.warmup + self.ramp:
            return self.peak * (e - self.warmup + 1) / self.ramp
        return self.peak

    def on_epoch_begin(self, epoch, logs=None):
        a = self.value(epoch)
        self.model.alpha.assign(np.float32(a))
        print("[alpha] ep %d = %.3f" % (epoch + 1, a))


class TimeGuard(keras.callbacks.Callback):
    def __init__(self, max_hours):
        super().__init__()
        self.deadline = time.time() + max_hours * 3600

    def on_epoch_end(self, epoch, logs=None):
        left = (self.deadline - time.time()) / 3600.0
        if left <= 0:
            print("[TimeGuard] time budget reached -> stop (best weights kept)")
            self.model.stop_training = True
        else:
            print("[TimeGuard] %.2fh left" % left)


class LrLog(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.lrs.append(float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)))
        except Exception:
            self.lrs.append(float("nan"))
