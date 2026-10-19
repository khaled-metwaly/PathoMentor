import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_CLASS_WEIGHTS = [0.0, 0.636, 0.674, 1.229, 1.461]
PATIENT_REGEX = r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})"


@dataclass
class TrainConfig:
    # what to run
    arm: str = "baseline"                 # baseline | mentor
    fov: int = 448                        # tile side in px at 0.5 MPP (448 or 256 in the paper)
    seed: int = 0

    # data
    train_dir: Optional[str] = None       # <dir>/image/*.png + <dir>/mask/*.png
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    class_weights: Optional[str] = None   # class_weights.json; auto-detected next to train_dir if omitted
    teacher_cache: Optional[str] = None   # uni2h_native_softmaps_32x32x5.npy (mentor arm only)
    teacher_index: Optional[str] = None   # uni2h_native_index.json (defaults to the cache's folder)
    patient_regex: str = PATIENT_REGEX
    out_dir: str = "runs"
    run_name: Optional[str] = None

    # optimisation (identical to the paper's notebooks)
    eff_batch: int = 32
    phys_batch: int = 4
    epochs: int = 25
    lr: float = 1e-4
    patience: int = 8
    rlr_patience: int = 4
    rlr_factor: float = 0.7
    min_lr: float = 1e-7
    monitor: str = "val_fwiou"
    ce_weight: float = 0.6
    dice_weight: float = 0.4
    augment: bool = True
    backbone_weights: str = "imagenet"    # imagenet | none

    # distillation (mentor arm)
    teach_grid: int = 32
    alpha_peak: float = 0.5
    align_warmup: int = 0
    align_ramp: int = 2
    kd_temp: float = 1.0

    # run hygiene
    max_hours: float = 11.0
    deterministic: bool = True
    max_train_tiles: Optional[int] = None   # debugging / smoke tests only

    @property
    def accum_steps(self):
        return max(1, self.eff_batch // self.phys_batch)

    @property
    def tag(self):
        return self.run_name or f"{self.arm}_fov{self.fov}_seed{self.seed}"

    def resolve(self):
        if self.arm not in ("baseline", "mentor"):
            raise ValueError("--arm must be 'baseline' or 'mentor'")
        if self.fov % self.teach_grid != 0 or self.fov % 32 != 0:
            raise ValueError(f"--fov must be a multiple of 32 and of teach_grid ({self.teach_grid}); got {self.fov}")
        for k in ("train_dir", "val_dir", "test_dir"):
            v = getattr(self, k)
            if v is None:
                raise ValueError(f"--{k.replace('_', '-')} is required")
            for sub in ("image", "mask"):
                if not os.path.isdir(os.path.join(v, sub)):
                    raise FileNotFoundError(f"{k}: expected sub-folder '{sub}/' in {v}")
        if self.arm == "mentor":
            if not self.teacher_cache or not os.path.isfile(self.teacher_cache):
                raise FileNotFoundError("--teacher-cache (uni2h_native_softmaps_32x32x5.npy) is required for --arm mentor")
            if not self.teacher_index:
                self.teacher_index = os.path.join(os.path.dirname(self.teacher_cache), "uni2h_native_index.json")
            if not os.path.isfile(self.teacher_index):
                raise FileNotFoundError(f"teacher index not found: {self.teacher_index}")
        if self.class_weights is None:
            for cand in (os.path.join(os.path.dirname(os.path.abspath(self.train_dir)), "class_weights.json"),
                         os.path.join(self.train_dir, "class_weights.json")):
                if os.path.isfile(cand):
                    self.class_weights = cand
                    break
        return self

    def load_class_weights(self):
        import numpy as np

        w = DEFAULT_CLASS_WEIGHTS
        if self.class_weights and os.path.isfile(self.class_weights):
            j = json.load(open(self.class_weights))
            w = j["weights"] if isinstance(j, dict) and "weights" in j else j
        w = np.array(w, np.float32)
        w[0] = 0.0
        return w

    def to_dict(self):
        return dataclasses.asdict(self)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def load_yaml(path):
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def build_parser(description="Train one PathoMentor run (one arm x one FOV x one seed)."):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = TrainConfig()
    p.add_argument("--config", help="YAML file with any of the options below (CLI overrides YAML)")
    g = p.add_argument_group("run")
    g.add_argument("--arm", choices=["baseline", "mentor"], default=None, help=f"default {d.arm}")
    g.add_argument("--fov", type=int, default=None, help=f"tile size in px at 0.5 MPP (paper: 448 or 256); default {d.fov}")
    g.add_argument("--seed", type=int, default=None, help=f"default {d.seed}")
    g.add_argument("--run-name", default=None, help="output tag (default <arm>_fov<fov>_seed<seed>)")
    g.add_argument("--out-dir", default=None, help=f"default {d.out_dir}")
    g = p.add_argument_group("data")
    g.add_argument("--data-root", default=None, help="folder with train/ val/ test/ (sets the three dirs below)")
    g.add_argument("--train-dir", default=None)
    g.add_argument("--val-dir", default=None)
    g.add_argument("--test-dir", default=None)
    g.add_argument("--class-weights", default=None, help="class_weights.json (auto-detected next to train dir)")
    g.add_argument("--teacher-cache", default=None, help="uni2h_native_softmaps_32x32x5.npy (mentor arm)")
    g.add_argument("--teacher-index", default=None, help="uni2h_native_index.json (default: next to the cache)")
    g.add_argument("--patient-regex", default=None, help="regex extracting the patient id from tile names")
    g.add_argument("--max-train-tiles", type=int, default=None, help="subsample train tiles (debug only)")
    g = p.add_argument_group("optimisation")
    for name, typ in [("eff_batch", int), ("phys_batch", int), ("epochs", int), ("lr", float), ("patience", int),
                      ("rlr_patience", int), ("rlr_factor", float), ("min_lr", float), ("ce_weight", float),
                      ("dice_weight", float)]:
        g.add_argument("--" + name.replace("_", "-"), type=typ, default=None, help=f"default {getattr(d, name)}")
    g.add_argument("--monitor", default=None, choices=["val_fwiou", "val_clinical_miou"], help=f"default {d.monitor}")
    g.add_argument("--augment", type=_str2bool, default=None, help=f"default {d.augment}")
    g.add_argument("--backbone-weights", choices=["imagenet", "none"], default=None, help=f"default {d.backbone_weights}")
    g = p.add_argument_group("distillation (mentor arm)")
    for name, typ in [("teach_grid", int), ("alpha_peak", float), ("align_warmup", int), ("align_ramp", int), ("kd_temp", float)]:
        g.add_argument("--" + name.replace("_", "-"), type=typ, default=None, help=f"default {getattr(d, name)}")
    g = p.add_argument_group("hygiene")
    g.add_argument("--max-hours", type=float, default=None, help=f"time guard; default {d.max_hours}")
    g.add_argument("--deterministic", type=_str2bool, default=None, help=f"default {d.deterministic}")
    return p


def config_from_args(args):
    values = {}
    if getattr(args, "config", None):
        values.update({k.replace("-", "_"): v for k, v in load_yaml(args.config).items()})
    for k, v in vars(args).items():
        if k in ("config", "data_root") or v is None:
            continue
        values[k] = v
    root = values.pop("data_root", None)
    if getattr(args, "data_root", None):
        root = args.data_root
    if root:
        for s in ("train", "val", "test"):
            values.setdefault(f"{s}_dir", os.path.join(root, s))
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**values)
