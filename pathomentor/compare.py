import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

METRICS = ["macro", "weighted", "micro"]
CLASSES = [("tumor", "tumor"), ("stroma", "stroma"), ("inflam", "inflammatory"), ("necro", "necrosis")]
ARM_NAME = {"baseline": "Baseline", "mentor": "PathoMentor"}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def discover(runs_dir):
    runs = {}
    for rp in sorted(glob.glob(os.path.join(runs_dir, "**", "*_results.json"), recursive=True)):
        r = json.load(open(rp))
        key = (r["arm"], int(r.get("fov", r.get("image_size"))), int(r["seed"]))
        pp = rp.replace("_results.json", "_per_patient.csv")
        hist = rp.replace("_results.json", "_history.csv")
        if key in runs:
            print("[warn] duplicate run", key, "->", rp, "(keeping the first)")
            continue
        runs[key] = dict(results=r, per_patient=_read_pp(pp) if os.path.isfile(pp) else {},
                         history=hist if os.path.isfile(hist) else None, path=rp)
    return runs


def _read_pp(path):
    out = {}
    for row in csv.DictReader(open(path)):
        d = {}
        for k, v in row.items():
            if k in ("patient", "arm", "present"):
                continue
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = float("nan")
        out[row.get("patient") or row.get("fov")] = d
    return out


def seed_avg(runs, arm, fov, seeds, key, require_gt=None):
    per = defaultdict(list)
    for s in seeds:
        for p, d in runs[(arm, fov, s)]["per_patient"].items():
            v = d.get(key, float("nan"))
            if require_gt and not d.get(require_gt, 0) > 0:
                continue
            if np.isfinite(v):
                per[p].append(v)
    return {p: float(np.mean(v)) for p, v in per.items()}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def boot_ci(d, stat=np.mean, B=20000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.RandomState(seed)
    n = len(d)
    bs = np.array([stat(d[rng.randint(0, n, n)]) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def trim20(d):
    return float(stats.trim_mean(d, 0.2))


def sign_test(d):
    d = np.asarray(d)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    p = float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else float("nan")
    return p, pos, neg


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def battery(d):
    # paired per-patient differences (pp); robust = Wilcoxon p<0.05 AND bootstrap CI excludes 0
    d = np.asarray(d, float)
    out = dict(n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), trim20=trim20(d))
    if len(d) >= 3 and np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
        out["t_p"] = float(stats.ttest_1samp(d, 0).pvalue)
        out["d_z"] = float(d.mean() / (d.std(ddof=1) + 1e-12))
    else:
        out.update(wilcoxon_p=float("nan"), t_p=float("nan"), d_z=float("nan"))
    out["sign_p"], out["better"], out["worse"] = sign_test(d)
    lo, hi, bp = boot_ci(d)
    out["ci95"], out["boot_p"] = [lo, hi], bp
    tlo, thi, _ = boot_ci(d, trim20)
    out["trim_ci95"] = [tlo, thi]
    out["robust"] = bool(out["wilcoxon_p"] < 0.05 and (lo > 0 or hi < 0))
    return out


def paired(a, b):
    ff = sorted(set(a) & set(b))
    return ff, np.array([(b[f] - a[f]) * 100 for f in ff], float)


# ---------------------------------------------------------------------------
# within-FOV: PathoMentor vs Baseline
# ---------------------------------------------------------------------------
def compare_fov(runs, fov, seeds):
    rep = dict(fov=fov, seeds=seeds, per_patient={}, per_class=[], seed_level={}, best_epoch={})
    for m in METRICS:
        B = seed_avg(runs, "baseline", fov, seeds, m)
        M = seed_avg(runs, "mentor", fov, seeds, m)
        ff, d = paired(B, M)
        r = battery(d)
        r.update(baseline=float(np.mean([B[f] for f in ff]) * 100), mentor=float(np.mean([M[f] for f in ff]) * 100))
        rep["per_patient"][m] = r
    rows = []
    for sc, disp in CLASSES:
        B = seed_avg(runs, "baseline", fov, seeds, "iou_" + sc, require_gt="gt_" + sc)
        M = seed_avg(runs, "mentor", fov, seeds, "iou_" + sc, require_gt="gt_" + sc)
        ff, d = paired(B, M)
        if len(d) < 5:
            continue
        r = battery(d)
        r.update(cls=disp, baseline=float(np.mean([B[f] for f in ff]) * 100), mentor=float(np.mean([M[f] for f in ff]) * 100))
        rows.append(r)
    for r, a in zip(rows, holm([r["wilcoxon_p"] for r in rows]) if rows else []):
        r["holm_p"] = float(a)
        r["robust"] = bool(a < 0.05 and (r["ci95"][0] > 0 or r["ci95"][1] < 0))
    rep["per_class"] = rows
    for m in METRICS:
        bv = [runs[("baseline", fov, s)]["results"]["test"][m] * 100 for s in seeds]
        mv = [runs[("mentor", fov, s)]["results"]["test"][m] * 100 for s in seeds]
        dd = np.array(mv) - np.array(bv)
        rep["seed_level"][m] = dict(baseline=bv, mentor=mv, per_seed_delta=dd.round(3).tolist(), delta=float(dd.mean()),
                                    sd_baseline=float(np.std(bv, ddof=1)) if len(bv) > 1 else float("nan"),
                                    sd_mentor=float(np.std(mv, ddof=1)) if len(mv) > 1 else float("nan"),
                                    t_p=float(stats.ttest_1samp(dd, 0).pvalue) if len(dd) > 1 else float("nan"))
    for a in ("baseline", "mentor"):
        rep["best_epoch"][a] = [runs[(a, fov, s)]["results"].get("val_best_epoch") for s in seeds]
        rep["best_epoch"][a + "_converged"] = [runs[(a, fov, s)]["results"].get("converged_ok") for s in seeds]
    return rep


def print_fov(rep):
    print("\n" + "=" * 92)
    print("FOV %d px | seeds %s | per-patient, seed-averaged then paired" % (rep["fov"], rep["seeds"]))
    print("=" * 92)
    print("metric    |  n | Baseline PathoMentor |  delta  [95% CI]          | Wilcoxon   sign    t      | d_z  | +/-")
    for m in METRICS:
        r = rep["per_patient"][m]
        print("%-9s | %2d |  %6.2f    %6.2f    | %+5.2f [%+5.2f,%+5.2f]   | %8.2g %7.2g %7.2g | %4.2f | %d/%d" % (
            m, r["n"], r["baseline"], r["mentor"], r["mean"], r["ci95"][0], r["ci95"][1],
            r["wilcoxon_p"], r["sign_p"], r["t_p"], r["d_z"], r["better"], r["worse"]))
    print("\nper class (present-only) | n | Baseline PathoMentor | delta [95% CI] | Holm p | robust")
    for r in rep["per_class"]:
        print("  %-12s %2d | %6.2f %6.2f | %+5.2f [%+5.2f,%+5.2f] | %8.2g | %s" % (
            r["cls"], r["n"], r["baseline"], r["mentor"], r["mean"], r["ci95"][0], r["ci95"][1], r["holm_p"], r["robust"]))
    print("\nseed level (pooled test, n=%d seeds):" % len(rep["seeds"]))
    for m in METRICS:
        s = rep["seed_level"][m]
        print("  %-9s per-seed delta %s | mean %+.2f | t p=%.3g | SD Baseline %.2f / PathoMentor %.2f" % (
            m, s["per_seed_delta"], s["delta"], s["t_p"], s["sd_baseline"], s["sd_mentor"]))
    print("best epochs: Baseline %s | PathoMentor %s" % (rep["best_epoch"]["baseline"], rep["best_epoch"]["mentor"]))


# ---------------------------------------------------------------------------
# cross-FOV (same patients): the paper's FOV-vs-distillation facts
# ---------------------------------------------------------------------------
def compare_cross_fov(runs, big, small, seeds_big, seeds_small, metric="weighted"):
    Bb = seed_avg(runs, "baseline", big, seeds_big, metric)
    Mb = seed_avg(runs, "mentor", big, seeds_big, metric)
    Bs = seed_avg(runs, "baseline", small, seeds_small, metric)
    Ms = seed_avg(runs, "mentor", small, seeds_small, metric)
    ff = sorted(set(Bb) & set(Mb) & set(Bs) & set(Ms))
    g_big = np.array([Mb[f] - Bb[f] for f in ff]) * 100
    g_small = np.array([Ms[f] - Bs[f] for f in ff]) * 100
    out = dict(metric=metric, big=big, small=small, n=len(ff))
    out["gain_small_minus_gain_big"] = battery(g_small - g_big)
    out["fov_lever_baseline_big_minus_small"] = battery(np.array([Bb[f] - Bs[f] for f in ff]) * 100)
    out["mentor_small_minus_baseline_big"] = battery(np.array([Ms[f] - Bb[f] for f in ff]) * 100)
    out["mentor_small_minus_mentor_big"] = battery(np.array([Ms[f] - Mb[f] for f in ff]) * 100)
    return out


def print_cross(c):
    print("\n" + "=" * 92)
    print("CROSS-FOV on the same %d patients (%s) - %d px vs %d px" % (c["n"], c["metric"], c["big"], c["small"]))
    print("=" * 92)
    for k, lab in [("gain_small_minus_gain_big", "PathoMentor gain @%d - gain @%d" % (c["small"], c["big"])),
                   ("fov_lever_baseline_big_minus_small", "FOV lever: Baseline @%d - Baseline @%d" % (c["big"], c["small"])),
                   ("mentor_small_minus_baseline_big", "PathoMentor @%d - Baseline @%d" % (c["small"], c["big"])),
                   ("mentor_small_minus_mentor_big", "PathoMentor @%d - PathoMentor @%d" % (c["small"], c["big"]))]:
        r = c[k]
        print("  %-42s %+5.2f [%+5.2f,%+5.2f]  Wilcoxon p=%.2g  %d/%d" % (
            lab, r["mean"], r["ci95"][0], r["ci95"][1], r["wilcoxon_p"], r["better"], r["n"]))


# ---------------------------------------------------------------------------
# figures (labelled by FOV and arm only)
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    return plt


def fig_confusion(runs, fov, seeds, path):
    plt = _plt()
    names = ["tumour", "stroma", "inflam.", "necrosis"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, arm in zip(ax, ["baseline", "mentor"]):
        cm = np.sum([np.array(runs[(arm, fov, s)]["results"]["test"]["confusion_matrix"], float) for s in seeds], axis=0)
        sub = cm[1:, 1:]
        n = sub / (sub.sum(1, keepdims=True) + 1e-9)
        a.imshow(n, vmin=0, vmax=1, cmap="Blues")
        a.set_xticks(range(4))
        a.set_xticklabels(names, rotation=30)
        a.set_yticks(range(4))
        a.set_yticklabels(names)
        a.set_xlabel("predicted")
        a.set_ylabel("ground truth")
        a.set_title("%s, %d px" % (ARM_NAME[arm], fov))
        for i in range(4):
            for j in range(4):
                a.text(j, i, "%.2f" % n[i, j], ha="center", va="center", fontsize=9, color="white" if n[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_curves(runs, fov, seeds, path, alpha_peak=0.5, ramp=2):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    style = {"baseline": dict(color="#D55E00", ls="--"), "mentor": dict(color="#0072B2", ls="-")}
    maxep = 1
    for arm in ["baseline", "mentor"]:
        curves = []
        for s in seeds:
            hp = runs[(arm, fov, s)]["history"]
            if not hp:
                continue
            rows = list(csv.DictReader(open(hp)))
            fw = np.array([float(r["val_fwiou"]) * 100 for r in rows])
            curves.append(fw)
            maxep = max(maxep, len(fw))
            ax.plot(range(1, len(fw) + 1), fw, lw=0.8, alpha=0.35, **style[arm])
        if curves:
            L = max(len(c) for c in curves)
            mat = np.full((len(curves), L), np.nan)
            for i, c in enumerate(curves):
                mat[i, :len(c)] = c
            ax.plot(range(1, L + 1), np.nanmean(mat, 0), lw=2.4, label="%s (seed mean)" % ARM_NAME[arm], **style[arm])
    ax.axvspan(0.5, ramp + 0.5, color="#F0E442", alpha=0.25, label="alpha warm-up 0 -> %.1f" % alpha_peak)
    ax.set_xlim(0.5, maxep + 0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation FWIoU (%)")
    ax.set_title("Validation FWIoU, %d px (thin = individual seeds)" % fov)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_box(rep, runs, fov, seeds, path):
    plt = _plt()
    data, labs = [], []
    for m, lab in zip(METRICS, ["macro (ClinMIoU)", "weighted (FWIoU)", "micro"]):
        _, d = paired(seed_avg(runs, "baseline", fov, seeds, m), seed_avg(runs, "mentor", fov, seeds, m))
        data.append(d)
        labs.append("%s\np=%.2g" % (lab, rep["per_patient"][m]["wilcoxon_p"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot(data, showmeans=True)
    ax.axhline(0, color="#D55E00", lw=1)
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels(labs)
    ax.set_ylabel("per-patient PathoMentor - Baseline (pp)")
    ax.set_title("%d px" % fov)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_per_patient(runs, fov, seeds, path):
    SA = {(a, m): seed_avg(runs, a, fov, seeds, m) for a in ("baseline", "mentor") for m in METRICS}
    pats = sorted(set(SA[("baseline", "weighted")]) & set(SA[("mentor", "weighted")]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient"] + ["%s_%s" % (a, m) for a in ("baseline", "mentor") for m in METRICS] + ["delta_%s" % m for m in METRICS])
        for p in pats:
            row = [p] + ["%.4f" % (SA[(a, m)][p] * 100) for a in ("baseline", "mentor") for m in METRICS]
            row += ["%.4f" % ((SA[("mentor", m)][p] - SA[("baseline", m)][p]) * 100) for m in METRICS]
            w.writerow(row)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Paired PathoMentor-vs-Baseline statistics over seeds (and across FOVs).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--runs", required=True, help="folder containing run outputs (searched recursively for *_results.json)")
    p.add_argument("--fov", type=int, nargs="*", default=None, help="FOVs to analyse (default: all found)")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="seeds to use (default: all seeds present for BOTH arms)")
    p.add_argument("--out-dir", default="compare")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)

    runs = discover(a.runs)
    if not runs:
        raise SystemExit("no *_results.json found under " + a.runs)
    fovs = sorted(set(k[1] for k in runs), reverse=True) if not a.fov else a.fov
    os.makedirs(a.out_dir, exist_ok=True)
    summary, seeds_by_fov = {}, {}
    for fov in fovs:
        sb = {k[2] for k in runs if k[0] == "baseline" and k[1] == fov}
        sm = {k[2] for k in runs if k[0] == "mentor" and k[1] == fov}
        requested = set(a.seeds) if a.seeds else (sb | sm)
        seeds = sorted(requested & sb & sm)
        unmatched = sorted(requested - set(seeds))
        if unmatched:
            print("[warn] FOV %d: seeds without both arms are skipped (balanced design): %s" % (fov, unmatched))
        if len(seeds) < 1:
            print("[skip] FOV %d: no seed has both arms" % fov)
            continue
        rep = compare_fov(runs, fov, seeds)
        print_fov(rep)
        summary["fov%d" % fov] = rep
        seeds_by_fov[fov] = seeds
        write_per_patient(runs, fov, seeds, os.path.join(a.out_dir, "fov%d_per_patient_deltas.csv" % fov))
        if not a.no_figures:
            fig_confusion(runs, fov, seeds, os.path.join(a.out_dir, "fov%d_confusion.png" % fov))
            fig_curves(runs, fov, seeds, os.path.join(a.out_dir, "fov%d_training_curves.png" % fov))
            fig_box(rep, runs, fov, seeds, os.path.join(a.out_dir, "fov%d_delta_box.png" % fov))
    done = sorted(seeds_by_fov, reverse=True)
    if len(done) >= 2:
        big, small = done[0], done[-1]
        summary["cross_fov"] = {}
        for m in METRICS:
            c = compare_cross_fov(runs, big, small, seeds_by_fov[big], seeds_by_fov[small], m)
            summary["cross_fov"][m] = c
            if m == "weighted":
                print_cross(c)
    json.dump(summary, open(os.path.join(a.out_dir, "compare_stats.json"), "w"), indent=2, default=float)
    print("\nwrote", a.out_dir)


if __name__ == "__main__":
    main()
