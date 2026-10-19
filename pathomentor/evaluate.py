import argparse
import csv
import json
import os
from collections import defaultdict

from .determinism import set_deterministic_env

set_deterministic_env()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from . import NUM_CLASSES  # noqa: E402
from .callbacks import predict_logits  # noqa: E402
from .config import PATIENT_REGEX  # noqa: E402
from .data import build_eval, patient_of  # noqa: E402
from .metrics import boundary_collar, clinical_metrics, confusion  # noqa: E402


def evaluate_split(model, split_dir, image_size, batch_size=4, patient_regex=PATIENT_REGEX, boundary_ks=()):
    # One pass over the split -> pooled confusion, per-patient confusions, optional boundary-tolerant confusions.
    ds, paths = build_eval(split_dir, image_size, batch_size, with_paths=True)
    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    per_patient = defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64))
    bnd = {int(k): np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64) for k in boundary_ks}
    i = 0
    for imgs, masks in ds:
        pr = tf.argmax(predict_logits(model, imgs), -1).numpy()
        gt = tf.cast(tf.squeeze(masks, -1), tf.int32).numpy()
        for j in range(pr.shape[0]):
            cm = confusion(gt[j], pr[j])
            pooled += cm
            per_patient[patient_of(paths[i], patient_regex)] += cm
            for k in bnd:
                bnd[k] += confusion(gt[j], pr[j], valid=~boundary_collar(gt[j], k))
            i += 1
    return pooled, dict(per_patient), bnd


def summarise(pooled, per_patient):
    m = clinical_metrics(pooled)
    acc = float(np.trace(pooled) / (pooled.sum() + 1e-6))
    iou_all = [float(pooled[c, c] / (pooled[c, :].sum() + pooled[:, c].sum() - pooled[c, c] + 1e-6)) for c in range(NUM_CLASSES)]
    test = dict(accuracy=acc, macro=m["macro"], weighted=m["weighted"], micro=m["micro"], clinical_miou=m["macro"],
                per_class_iou=m["iou"], gt_clin=m["gt"], present=m["present"], iou_all5=iou_all,
                confusion_matrix=pooled.tolist(), n_test_patients=len(per_patient))
    pp = {p: clinical_metrics(cm) for p, cm in sorted(per_patient.items())}
    return test, pp


def write_per_patient_csv(path, pp, arm, fov, seed):
    def gv(v):
        return ("%.6f" % v) if (isinstance(v, float) and np.isfinite(v)) else ""

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient", "arm", "fov", "seed", "macro", "weighted", "micro",
                    "iou_tumor", "iou_stroma", "iou_inflam", "iou_necro",
                    "gt_tumor", "gt_stroma", "gt_inflam", "gt_necro", "present"])
        for p, d in pp.items():
            w.writerow([p, arm, fov, seed, gv(d["macro"]), gv(d["weighted"]), gv(d["micro"]),
                        gv(d["iou"]["tumor"]), gv(d["iou"]["stroma"]), gv(d["iou"]["inflam"]), gv(d["iou"]["necro"]),
                        d["gt"]["tumor"], d["gt"]["stroma"], d["gt"]["inflam"], d["gt"]["necro"], "|".join(d["present"])])


def print_summary(label, test):
    print("=== %s (pooled test) ===" % label)
    print("  Acc %.2f | MACRO %.2f | FWIoU %.2f | MICRO %.2f | patients %d" % (
        test["accuracy"] * 100, test["macro"] * 100, test["weighted"] * 100, test["micro"] * 100, test["n_test_patients"]))
    print("  per-class IoU: " + "  ".join("%s=%.1f" % (k, v * 100) for k, v in test["per_class_iou"].items()))


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate saved PathoMentor / baseline weights on a test split.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True, help="*.weights.h5 written by pathomentor.train")
    p.add_argument("--fov", type=int, required=True, help="tile size the weights were trained at (448 / 256)")
    p.add_argument("--test-dir", required=True, help="<dir>/image + <dir>/mask")
    p.add_argument("--out-dir", default="eval")
    p.add_argument("--label", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patient-regex", default=PATIENT_REGEX)
    p.add_argument("--boundary-k", type=int, nargs="*", default=[],
                   help="also report boundary-tolerant IoU ignoring a k-px collar around GT borders, e.g. 0 4 8 16")
    a = p.parse_args(argv)

    from .model import build_deeplabv3plus

    model = build_deeplabv3plus(image_size=a.fov, imagenet=False)
    model.load_weights(a.weights)
    label = a.label or os.path.basename(a.weights).replace(".weights.h5", "")
    pooled, per_patient, bnd = evaluate_split(model, a.test_dir, a.fov, a.batch_size, a.patient_regex, a.boundary_k)
    test, pp = summarise(pooled, per_patient)
    print_summary(label, test)
    os.makedirs(a.out_dir, exist_ok=True)
    out = dict(label=label, weights=a.weights, fov=a.fov, test=test)
    if bnd:
        out["boundary"] = {}
        for k, cm in sorted(bnd.items()):
            m = clinical_metrics(cm)
            out["boundary"]["k%d" % k] = dict(macro=m["macro"], weighted=m["weighted"], micro=m["micro"],
                                              per_class_iou=m["iou"], kept_fraction=float(cm.sum() / max(pooled.sum(), 1)))
            print("  boundary k=%2dpx: macro %.2f | FWIoU %.2f | kept %.1f%% of clinical px" % (
                k, m["macro"] * 100, m["weighted"] * 100, 100 * cm.sum() / max(pooled.sum(), 1)))
    json.dump(out, open(os.path.join(a.out_dir, label + "_eval.json"), "w"), indent=2)
    write_per_patient_csv(os.path.join(a.out_dir, label + "_per_patient.csv"), pp, "", a.fov, "")
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
