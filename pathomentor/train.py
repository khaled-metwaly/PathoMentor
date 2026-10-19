import csv
import gc
import json
import os
import sys
import time

from .determinism import set_deterministic_env

set_deterministic_env()

import numpy as np  # noqa: E402
from tensorflow import keras  # noqa: E402

from . import __version__  # noqa: E402
from .callbacks import AlphaSchedule, LrLog, SegMetrics, TimeGuard  # noqa: E402
from .config import build_parser, config_from_args  # noqa: E402
from .data import TeacherCache, build_eval, build_train  # noqa: E402
from .determinism import enable_tf_determinism, seed_everything  # noqa: E402
from .evaluate import evaluate_split, print_summary, summarise, write_per_patient_csv  # noqa: E402
from .losses import make_seg_loss  # noqa: E402
from .model import DistillModel, build_deeplabv3plus, make_optimizer  # noqa: E402


def run(cfg):
    cfg.resolve()
    deterministic = enable_tf_determinism() if cfg.deterministic else False
    out = os.path.join(cfg.out_dir, cfg.tag)
    os.makedirs(out, exist_ok=True)
    json.dump(cfg.to_dict(), open(os.path.join(out, "config.json"), "w"), indent=2)
    print("PathoMentor %s | arm=%s fov=%d seed=%d | eff batch %d (%d x %d accum) | op-determinism %s" % (
        __version__, cfg.arm, cfg.fov, cfg.seed, cfg.eff_batch, cfg.phys_batch, cfg.accum_steps, deterministic))

    class_weights = cfg.load_class_weights()
    print("class weights", np.round(class_weights, 4), "| from", cfg.class_weights or "defaults")
    seg_loss = make_seg_loss(class_weights, cfg.ce_weight, cfg.dice_weight)

    keras.backend.clear_session()
    seed_everything(cfg.seed)   # identical init for baseline and mentor at the same seed
    teacher = TeacherCache(cfg.teacher_cache, cfg.teacher_index, cfg.teach_grid) if cfg.arm == "mentor" else None
    train_ds, n_train = build_train(cfg, cfg.seed, teacher)
    val_ds = build_eval(cfg.val_dir, cfg.fov, cfg.phys_batch)
    print("train tiles", n_train)

    opt = make_optimizer(cfg.lr, cfg.accum_steps)
    base = build_deeplabv3plus(image_size=cfg.fov, imagenet=(cfg.backbone_weights == "imagenet"))
    seg_metrics = SegMetrics(val_ds)
    lrlog = LrLog()
    es = keras.callbacks.EarlyStopping(monitor=cfg.monitor, mode="max", patience=cfg.patience,
                                       restore_best_weights=True, verbose=1)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor=cfg.monitor, mode="max", factor=cfg.rlr_factor,
                                            patience=cfg.rlr_patience, min_lr=cfg.min_lr, verbose=1)
    if cfg.arm == "mentor":
        model = DistillModel(base, seg_loss, cfg.fov, cfg.teach_grid, cfg.kd_temp)
        _ = model(np.zeros([1, cfg.fov, cfg.fov, 3], np.float32))
        model.compile(optimizer=opt)
        callbacks = [TimeGuard(cfg.max_hours), seg_metrics,
                     AlphaSchedule(cfg.alpha_peak, cfg.align_warmup, cfg.align_ramp), lrlog, es, rlr]
    else:
        model = base
        model.compile(optimizer=opt, loss=seg_loss)
        callbacks = [TimeGuard(cfg.max_hours), seg_metrics, lrlog, es, rlr]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=2)
    h = seg_metrics.hist
    b = int(np.argmax(h["val_fwiou" if cfg.monitor == "val_fwiou" else "val_clinical_miou"]))
    conv_ok = bool(b + 1 <= cfg.epochs - 2)
    print("done %.1f min | best epoch %d (val FWIoU %.2f) | converged_ok=%s" % (
        (time.time() - t0) / 60, b + 1, h["val_fwiou"][b] * 100, conv_ok))
    if not conv_ok:
        print("[WARN] best epoch is at the epoch ceiling -> run may be under-trained")

    weights_path = os.path.join(out, cfg.tag + ".weights.h5")
    base.save_weights(weights_path)   # deployed model = plain DeepLabV3+ (teacher not needed)

    pooled, per_patient, _ = evaluate_split(base, cfg.test_dir, cfg.fov, cfg.phys_batch, cfg.patient_regex)
    test, pp = summarise(pooled, per_patient)
    print_summary(cfg.tag, test)
    write_per_patient_csv(os.path.join(out, cfg.tag + "_per_patient.csv"), pp, cfg.arm, cfg.fov, cfg.seed)

    res = dict(version=__version__, arm=cfg.arm, fov=cfg.fov, seed=cfg.seed, deterministic=bool(deterministic),
               image_size=cfg.fov, epochs=cfg.epochs, epochs_run=len(h["val_fwiou"]), monitor=cfg.monitor,
               val_best_epoch=b + 1, converged_ok=conv_ok, n_train=n_train, train_minutes=(time.time() - t0) / 60,
               val=dict(fwiou=h["val_fwiou"][b], macro=h["val_clinical_miou"][b], clin3=h["val_clin3"][b]),
               test=test, weights_file=os.path.basename(weights_path))
    json.dump(res, open(os.path.join(out, cfg.tag + "_results.json"), "w"), indent=2)

    with open(os.path.join(out, cfg.tag + "_history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "val_fwiou", "val_macro", "val_clin3", "val_dice", "train_loss", "lr"])
        tl = history.history.get("loss", [])
        for e in range(len(h["val_fwiou"])):
            w.writerow([e + 1, round(h["val_fwiou"][e], 6), round(h["val_clinical_miou"][e], 6),
                        round(h["val_clin3"][e], 6), round(h["val_dice"][e], 6),
                        round(float(tl[e]), 6) if e < len(tl) else "",
                        lrlog.lrs[e] if e < len(lrlog.lrs) else ""])
    _plot_history(h, b, conv_ok, cfg, os.path.join(out, cfg.tag + "_history.png"))
    print("outputs ->", out)
    del model, train_ds, val_ds
    gc.collect()
    return res


def _plot_history(h, b, conv_ok, cfg, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm = "PathoMentor" if cfg.arm == "mentor" else "Baseline"
    ep = range(1, len(h["val_fwiou"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [v * 100 for v in h["val_fwiou"]], lw=2, label="val FWIoU (monitor)")
    plt.plot(ep, [v * 100 for v in h["val_clinical_miou"]], "--", label="val macro")
    plt.axvline(b + 1, color="r", ls=":")
    plt.title("%s, %d px, seed %d - best epoch %d%s" % (arm, cfg.fov, cfg.seed, b + 1, "" if conv_ok else " [at ceiling]"))
    plt.xlabel("epoch")
    plt.ylabel("%")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
        run(cfg)
    except (ValueError, FileNotFoundError) as e:
        print("error:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
