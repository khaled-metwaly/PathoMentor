import tensorflow as tf

from . import IGNORE_INDEX, NUM_CLASSES


def make_seg_loss(class_weights, ce_weight=0.6, dice_weight=0.4):
    # 0.6 * class-weighted CE + 0.4 * Dice over the clinical classes 1-4; background (0) ignored
    cw = tf.constant(class_weights, tf.float32)

    def _vs(y_true):
        y = tf.cast(tf.squeeze(y_true, -1), tf.int32)
        valid = tf.cast(tf.not_equal(y, IGNORE_INDEX), tf.float32)
        y_safe = tf.where(tf.equal(y, IGNORE_INDEX), tf.zeros_like(y), y)
        return y, valid, y_safe

    def weighted_ce(y_true, logits):
        _, valid, ys = _vs(y_true)
        pw = tf.gather(cw, ys) * valid
        ce = tf.keras.losses.sparse_categorical_crossentropy(ys, logits, from_logits=True)
        return tf.reduce_sum(ce * pw) / (tf.reduce_sum(pw) + 1e-6)

    def dice_loss(y_true, logits):
        _, valid, ys = _vs(y_true)
        probs = tf.nn.softmax(logits, -1)
        oh = tf.one_hot(ys, NUM_CLASSES) * tf.expand_dims(valid, -1)
        probs = probs * tf.expand_dims(valid, -1)
        num = 2.0 * tf.reduce_sum(oh * probs, axis=[1, 2])
        den = tf.reduce_sum(oh + probs, axis=[1, 2]) + 1e-6
        return tf.reduce_mean(1.0 - num[:, 1:] / den[:, 1:])

    def seg_loss(y_true, logits):
        return ce_weight * weighted_ce(y_true, logits) + dice_weight * dice_loss(y_true, logits)

    return seg_loss
