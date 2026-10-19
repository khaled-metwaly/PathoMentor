import os


def set_deterministic_env():
    # must run before TensorFlow is imported
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def enable_tf_determinism():
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    try:
        tf.config.experimental.enable_op_determinism()
        return True
    except Exception as e:
        print("op-determinism unavailable -> seed-only:", str(e)[:80])
        return False


def seed_everything(seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
