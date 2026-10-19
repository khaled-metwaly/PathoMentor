import numpy as np

from . import CLASS_SHORT, CLINICAL_CLASSES, IGNORE_INDEX, NUM_CLASSES


def confusion(gt, pred, valid=None):
    gt = np.asarray(gt, np.int64).reshape(-1)
    pred = np.asarray(pred, np.int64).reshape(-1)
    keep = gt != IGNORE_INDEX
    if valid is not None:
        keep &= np.asarray(valid, bool).reshape(-1)
    return np.bincount(NUM_CLASSES * gt[keep] + pred[keep], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def all_class_metrics(cm):
    # used by the validation callback (val_fwiou / val_clinical_miou)
    iou, dice, rec, prec, freq = [], [], [], [], []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou.append(tp / (tp + fp + fn + 1e-6))
        dice.append(2 * tp / (2 * tp + fp + fn + 1e-6))
        rec.append(tp / (cm[c, :].sum() + 1e-6))
        prec.append(tp / (cm[:, c].sum() + 1e-6))
        freq.append(cm[c, :].sum())
    freq = np.array(freq, np.float64)
    freq /= freq.sum() + 1e-6
    return iou, dice, rec, prec, freq


def clinical_metrics(cm):
    # macro (= ClinMIoU), weighted (= FWIoU) and micro (pooled-pixel) IoU over the 4 clinical classes,
    # all from ONE confusion matrix.  Absent classes are excluded from macro/weighted.
    cl = CLINICAL_CLASSES
    tp = np.array([cm[c, c] for c in cl], float)
    gt = np.array([cm[c, :].sum() for c in cl], float)
    pp = np.array([cm[:, c].sum() for c in cl], float)
    union = tp + (pp - tp) + (gt - tp)
    present = gt > 0
    iou = np.where(union > 0, tp / np.maximum(union, 1.0), np.nan)
    macro = float(np.nanmean(iou[present])) if present.any() else float("nan")
    if present.any() and gt[present].sum() > 0:
        wt = gt[present] / gt[present].sum()
        weighted = float(np.nansum(wt * iou[present]))
    else:
        weighted = float("nan")
    micro = float(tp.sum() / union.sum()) if union.sum() > 0 else float("nan")
    iouf = {CLASS_SHORT[c]: (float(iou[k]) if np.isfinite(iou[k]) else float("nan")) for k, c in enumerate(cl)}
    gtf = {CLASS_SHORT[c]: int(gt[k]) for k, c in enumerate(cl)}
    pres = [CLASS_SHORT[c] for k, c in enumerate(cl) if present[k]]
    return dict(macro=macro, weighted=weighted, micro=micro, iou=iouf, gt=gtf, present=pres)


def boundary_collar(gt, k):
    # k-px don't-care band around ground-truth class borders
    from scipy import ndimage

    b = np.zeros(gt.shape, bool)
    d = gt[:-1, :] != gt[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = gt[:, :-1] != gt[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return ndimage.binary_dilation(b, iterations=int(k)) if k > 0 else np.zeros(gt.shape, bool)
