import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, layers, models

from . import NUM_CLASSES


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=1, dilation_rate=1, **kw):
        super().__init__(**kw)
        self.conv = layers.Conv2D(filters, kernel_size, dilation_rate=dilation_rate,
                                  padding="same", use_bias=False, kernel_initializer="he_normal")
        self.bn = layers.BatchNormalization()

    def call(self, x, training=False):
        return tf.nn.relu(self.bn(self.conv(x), training=training))


class ASPP(layers.Layer):
    def __init__(self, filters=256, **kw):
        super().__init__(**kw)
        self.c1 = ConvBlock(filters, 1)
        self.c6 = ConvBlock(filters, 3, 6)
        self.c12 = ConvBlock(filters, 3, 12)
        self.c18 = ConvBlock(filters, 3, 18)
        self.gap = layers.GlobalAveragePooling2D()
        self.pc = ConvBlock(filters, 1)
        self.fuse = ConvBlock(filters, 1)

    def call(self, x, training=False):
        d = tf.shape(x)
        p = self.gap(x)
        p = tf.reshape(p, [-1, 1, 1, tf.shape(p)[-1]])
        p = self.pc(p, training=training)
        p = tf.image.resize(p, (d[1], d[2]))
        out = layers.concatenate([self.c1(x, training=training), self.c6(x, training=training),
                                  self.c12(x, training=training), self.c18(x, training=training), p])
        return self.fuse(out, training=training)


def build_deeplabv3plus(image_size=448, num_classes=NUM_CLASSES, imagenet=True):
    # DeepLabV3+ / ResNet-50 (output stride 16, low-level skip at stride 4), fine-tuned end to end.
    # Layer order is identical to the paper notebooks, so released .weights.h5 files load directly.
    inp = Input(shape=(image_size, image_size, 3))
    bb = keras.applications.ResNet50(weights="imagenet" if imagenet else None, include_top=False, input_tensor=inp)
    high = bb.get_layer("conv4_block6_2_relu").output
    low = bb.get_layer("conv2_block3_2_relu").output
    x = ASPP(256)(high)
    x = layers.UpSampling2D(size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]), interpolation="bilinear")(x)
    low = ConvBlock(48, 1)(low)
    x = layers.Concatenate()([x, low])
    x = ConvBlock(256, 3)(x)
    x = ConvBlock(256, 3)(x)
    x = layers.UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear")(x)
    logits = layers.Conv2D(num_classes, 1, padding="same")(x)
    return models.Model(inp, logits, name="DeepLabV3Plus")


class DistillModel(keras.Model):
    # PathoMentor training wrapper: segmentation loss + alpha * soft-CE to the cached UNI2-h
    # soft class-map at a 32x32 grid.  Only `base` is kept for inference.
    def __init__(self, base, seg_loss_fn, image_size, teach_grid=32, kd_temp=1.0, **kw):
        super().__init__(**kw)
        self.base = base
        self.seg_loss_fn = seg_loss_fn
        self.teach_grid = int(teach_grid)
        self.kd_temp = float(kd_temp)
        self.alpha = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.pool = layers.AveragePooling2D(pool_size=image_size // teach_grid)

    def call(self, x, training=False):
        return self.base(x, training=training)

    def kd_loss(self, y, logits, tsoft):
        ls = self.pool(logits)                                    # [B,32,32,5]
        logp = tf.nn.log_softmax(ls / self.kd_temp, -1)
        ce = -tf.reduce_sum(tsoft * logp, -1)                     # soft cross-entropy
        y32 = tf.image.resize(tf.cast(y, tf.float32), [self.teach_grid, self.teach_grid], method="nearest")[..., 0]
        valid = tf.cast(y32 > 0, tf.float32)                      # distil on clinical cells only
        return tf.reduce_sum(ce * valid) / (tf.reduce_sum(valid) + 1e-6)

    def train_step(self, data):
        x, (y, tsoft) = data
        with tf.GradientTape() as tape:
            logits = self.base(x, training=True)
            ls = self.seg_loss_fn(y, logits)
            lkd = self.kd_loss(y, logits, tsoft)
            lt = ls + self.alpha * lkd
        g = tape.gradient(lt, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        return {"loss": lt, "seg": ls, "kd": lkd, "alpha": tf.identity(self.alpha)}

    def test_step(self, data):
        x, y = data
        logits = self.base(x, training=False)
        return {"loss": self.seg_loss_fn(y, logits)}


def make_optimizer(lr, accum_steps):
    try:
        return keras.optimizers.Adam(lr, gradient_accumulation_steps=accum_steps)
    except TypeError:
        print("[warn] this Keras has no gradient_accumulation_steps -> effective batch = physical batch")
        return keras.optimizers.Adam(lr)
