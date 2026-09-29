#!/usr/bin/env python3
"""
Paired comparison of two models evaluated on the same MURA test set
(e.g. baseline vs. cast-augmented training), from the study-level predictions written by eval.py.

Outputs
  * paired DeLong test on the AUROC, paired (patient-cluster) bootstrap of the differences in
    AUROC / AUPRC / accuracy / F1 / Brier score, exact McNemar test, non-inferiority check
  * calibration: Brier score, ECE, calibration slope and intercept, reliability diagram
  * per body region and per class (sensitivity / specificity) comparison
  * optional: performance with vs. without a device (cast/splint) when an annotation file is given

Inputs (same files as eval.py / stats.ipynb):
  --labels     valid_labeled_studies.csv         (no header: study_path,label)
  --base       the *_mean.csv files from eval.py for model A (one per fold model; ensemble = mean)
  --mod        the *_mean.csv files from eval.py for model B
  --cast       optional CSV (header: study,cast) with cast in {0,1}; study = path as in the labels file
  --pred-col   column holding the study-level probability in the eval.py CSVs (default "0")

Everything is study-level (1 row = 1 MURA study). If the study path contains "patientXXXXX" the bootstrap
is resampled by patient (cluster bootstrap), the conservative choice when a patient has several studies.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (auc, brier_score_loss, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

RNG_SEED = 31


# ----------------------------------------------------------------------------------------------
# DeLong (Sun & Xu 2014, fast implementation) for two correlated AUROCs
# ----------------------------------------------------------------------------------------------
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(preds_sorted_T, m):
    """preds_sorted_T: (k, n) scores, positives first (first m columns). Returns aucs (k,), cov (k,k)."""
    n = preds_sorted_T.shape[1] - m
    pos, neg = preds_sorted_T[:, :m], preds_sorted_T[:, m:]
    k = preds_sorted_T.shape[0]
    tx, ty, tz = (np.empty([k, m]), np.empty([k, n]), np.empty([k, m + n]))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_sorted_T[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx, sy = np.cov(v01), np.cov(v10)
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([p1, p2])[:, order]
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), diff=float(diff),
                se=float(np.sqrt(max(var, 0))), z=float(z), p=float(p))


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def auprc(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    return auc(rc, pr)  # same definition as stats.ipynb


def metrics(y, p, thr=0.5):
    y = np.asarray(y)
    pred = p > thr
    return dict(AUROC=roc_auc_score(y, p), AUPRC=auprc(y, p), Accuracy=float((pred == y).mean()),
                F1=f1_score(y, pred))


def sens_at_spec(y, p, spec=0.90):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= (1 - spec)
    return float(tpr[ok].max()) if ok.any() else float("nan")


def wilson_or_cp(k, n, alpha=0.05):
    """Clopper-Pearson exact CI"""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    p = 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5).pvalue)
    return dict(a_only_correct=b, b_only_correct=c, p=p)


# ----------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_stats(y, p, n_bins=10):
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y)
    brier = brier_score_loss(y, p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf, acc = p[mask].mean(), y[mask].mean()
        ece += mask.mean() * abs(conf - acc)
        rows.append((float(conf), float(acc), int(mask.sum())))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])
    # calibration-in-the-large: intercept with slope fixed to 1 (Newton on offset-only logistic model)
    z, a = _logit(p), 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(z + a)))
        g, h = np.sum(y - q), np.sum(q * (1 - q))
        step = g / h if h > 0 else 0
        a += step
        if abs(step) < 1e-9:
            break
    return dict(brier=float(brier), ece=float(ece), slope=slope, intercept_citl=float(a), bins=rows)


def reliability_plot(y, p_base, p_mod, out_png, n_bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw=dict(width_ratios=[3, 2]))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for p, lab, col in ((p_base, "Original", "#1f77b4"), (p_mod, "Modified", "#d62728")):
        cs = calibration_stats(y, p, n_bins)
        xs = [r[0] for r in cs["bins"]]
        ys = [r[1] for r in cs["bins"]]
        ax[0].plot(xs, ys, "o-", color=col, label=f"{lab} (Brier {cs['brier']:.3f}, ECE {cs['ece']:.3f})")
        ax[1].hist(p, bins=20, range=(0, 1), alpha=0.5, color=col, label=lab)
    ax[0].set_xlabel("Mean predicted probability (abnormal)")
    ax[0].set_ylabel("Observed fraction abnormal")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("Reliability diagram (test set, study level)")
    ax[1].set_xlabel("Predicted probability")
    ax[1].set_ylabel("Studies")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Paired (cluster) bootstrap of deltas
# ----------------------------------------------------------------------------------------------
def paired_bootstrap(df, fn, n_boot=2000, cluster=None, seed=RNG_SEED):
    """fn(sub_df)->dict of scalars. Returns point estimate and percentile CI for every key."""
    rng = np.random.default_rng(seed)
    point = fn(df)
    if cluster is not None and df[cluster].nunique() < len(df):
        groups = [g.index.values for _, g in df.groupby(cluster)]
        draw = lambda: np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    else:
        draw = lambda: rng.integers(0, len(df), len(df))
    reps = {k: [] for k in point}
    for _ in range(n_boot):
        sub = df.loc[draw()]
        try:
            r = fn(sub)
        except ValueError:      # e.g. resample with a single class
            continue
        for k, v in r.items():
            reps[k].append(v)
    out = {}
    for k, v in reps.items():
        v = np.asarray(v, dtype=float)
        out[k] = dict(est=float(point[k]), lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)))
    return out


def delta_fn(thr=0.5):
    def fn(d):
        mb, mm = metrics(d.y.values, d.p_base.values, thr), metrics(d.y.values, d.p_mod.values, thr)
        out = {f"d{k}": mm[k] - mb[k] for k in mb}
        out["dBrier"] = brier_score_loss(d.y, d.p_mod) - brier_score_loss(d.y, d.p_base)
        return out
    return fn


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
def load_ensemble(files, n_expected, pred_col):
    cols = []
    for f in files:
        d = pd.read_csv(f)
        col = pred_col if pred_col in d.columns else d.columns[-1]
        cols.append(d[col].values.astype(float))
        if len(cols[-1]) != n_expected:
            sys.exit(f"{f}: {len(cols[-1])} predictions but {n_expected} labelled studies")
    return np.mean(cols, axis=0), np.stack(cols, axis=1)


def parse_meta(path):
    part = re.search(r"XR_([A-Z]+)", path)
    pat = re.search(r"(patient\d+)", path)
    return (part.group(1) if part else "UNK"), (pat.group(1) if pat else None)


def fmt(e, d=3):
    return f"{e['est']:.{d}f} ({e['lo']:.{d}f}, {e['hi']:.{d}f})"


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--mod", nargs="+", required=True)
    ap.add_argument("--cast", default=None)
    ap.add_argument("--pred-col", default="0")
    ap.add_argument("--out", default="compare_results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ni-margin", type=float, default=0.02,
                    help="non-inferiority margin on delta-AUROC (declare it a priori / justify it in the paper)")
    ap.add_argument("--min-group", type=int, default=10, help="min studies per class to report an AUROC in a subgroup")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(a.labels, header=None, names=["study", "y"])
    n = len(lab)
    p_base, _ = load_ensemble(a.base, n, a.pred_col)
    p_mod, _ = load_ensemble(a.mod, n, a.pred_col)
    df = lab.assign(p_base=p_base, p_mod=p_mod)
    meta = df.study.apply(parse_meta)
    df["part"] = [m[0] for m in meta]
    df["patient"] = [m[1] if m[1] else f"study{i}" for i, m in enumerate(meta)]
    df["cluster"] = df.patient
    if a.cast:
        c = pd.read_csv(a.cast)
        c.columns = ["study", "cast"] + list(c.columns[2:])
        exact = c.set_index("study")["cast"]
        df["cast"] = df.study.map(exact)
        if df.cast.isna().all():   # try substring key
            df["cast"] = df.study.apply(lambda s: next((v for k, v in exact.items() if k in s), np.nan))
        print(f"cast annotation matched for {df.cast.notna().sum()}/{n} studies")

    res, md = {}, []
    y = df.y.values.astype(int)

    # ---- 1. Overall discrimination + paired tests ---------------------------------------------
    dl = delong_test(y, df.p_mod.values, df.p_base.values)        # diff = modified - original
    boot = paired_bootstrap(df, delta_fn(), a.n_boot, "cluster")
    res["delong_modified_minus_original"] = dl
    res["paired_bootstrap"] = boot
    ni_ok = boot["dAUROC"]["lo"] > -a.ni_margin
    res["non_inferiority"] = dict(margin=a.ni_margin, lower95=boot["dAUROC"]["lo"], non_inferior=bool(ni_ok))
    mc = mcnemar_exact((df.p_base > 0.5).values == y, (df.p_mod > 0.5).values == y)
    res["mcnemar_accuracy_base_vs_mod"] = mc
    md += ["## Overall (test set, %d studies)" % n,
           f"- DeLong (modified vs original): AUROC modified {dl['auc_a']:.3f} vs original {dl['auc_b']:.3f}, "
           f"diff {dl['diff']:+.4f}, z={dl['z']:.2f}, **p" + ("<0.001" if dl['p'] < 0.001 else f"={dl['p']:.3f}") + "**",
           f"- Paired (cluster) bootstrap, {a.n_boot} resamples, delta = modified - original: "
           + "; ".join(f"{k} {fmt(v)}" for k, v in boot.items()),
           f"- Non-inferiority (margin {a.ni_margin}): lower 95% bound of dAUROC = {boot['dAUROC']['lo']:+.4f} -> "
           + ("non-inferior" if ni_ok else "NOT shown"),
           f"- McNemar exact on accuracy@0.5: {mc}",
           f"- Sensitivity at 90% specificity: original {sens_at_spec(y, df.p_base.values):.3f}, "
           f"modified {sens_at_spec(y, df.p_mod.values):.3f}", ""]
    print("\n".join(md))

    # ---- 2. Calibration -------------------------------------------------------------------------
    cal = {"original": calibration_stats(y, df.p_base.values), "modified": calibration_stats(y, df.p_mod.values)}
    res["calibration"] = cal
    reliability_plot(y, df.p_base.values, df.p_mod.values, out / "reliability.png")
    md += ["## Calibration (raw probabilities; models trained with pos_weight-weighted BCE, see note)",
           "| Model | Brier | ECE (10 bins) | Slope | Intercept (CITL) |", "|---|---|---|---|---|"]
    for k, v in cal.items():
        md.append(f"| {k} | {v['brier']:.3f} | {v['ece']:.3f} | {v['slope']:.2f} | {v['intercept_citl']:+.2f} |")
    md += [f"- Paired dBrier (mod - orig): {fmt(boot['dBrier'])}",
           "- NOTE: the weighted BCE re-balances the classes, so raw probabilities are not expected to match the "
           "MURA prevalence; discuss slope/ECE (and optionally a Platt recalibration fitted on validation folds).", ""]

    # ---- 3. Negative-transfer monitoring: per body part ------------------------------------------
    rows = []
    for part, d in df.groupby("part"):
        d = d.reset_index(drop=True)
        if d.y.nunique() < 2 or d.y.sum() < a.min_group or (1 - d.y).sum() < a.min_group:
            continue
        b = paired_bootstrap(d, lambda s: {"dAUROC": roc_auc_score(s.y, s.p_mod) - roc_auc_score(s.y, s.p_base)},
                             a.n_boot, "cluster")
        sens = lambda p, s=d: float(((p > 0.5) & (s.y == 1)).sum() / max((s.y == 1).sum(), 1))
        spec = lambda p, s=d: float(((p <= 0.5) & (s.y == 0)).sum() / max((s.y == 0).sum(), 1))
        rows.append(dict(part=part, n=len(d), n_abn=int(d.y.sum()),
                         auroc_base=roc_auc_score(d.y, d.p_base), auroc_mod=roc_auc_score(d.y, d.p_mod),
                         dAUROC=b["dAUROC"]["est"], lo=b["dAUROC"]["lo"], hi=b["dAUROC"]["hi"],
                         sens_base=sens(d.p_base), sens_mod=sens(d.p_mod), spec_base=spec(d.p_base),
                         spec_mod=spec(d.p_mod)))
    bp = pd.DataFrame(rows)
    res["by_body_part"] = rows
    bp.to_csv(out / "by_body_part.csv", index=False)
    md += ["## Per body part (negative-transfer check)", bp.round(3).to_markdown(index=False) if len(bp) else "n/a", ""]

    # ---- 4. Cast / splint stratified performance -----------------------------------------------
    if a.cast and df.cast.notna().any():
        d0 = df[df.cast.notna()].copy()
        d0["cast"] = d0.cast.astype(int)
        rows, tests = [], {}
        for cval, d in d0.groupby("cast"):
            for yv, name in ((0, "normal"), (1, "abnormal")):
                s = d[d.y == yv]
                if len(s) == 0:
                    continue
                for tag, col in (("base", "p_base"), ("mod", "p_mod")):
                    correct = ((s[col] > 0.5) == bool(yv))
                    k = int(correct.sum())
                    lo, hi = wilson_or_cp(k, len(s))
                    rows.append(dict(cast=cval, truth=name, model=tag, n=len(s), correct=k,
                                     rate=k / len(s), lo=lo, hi=hi))
                tests[f"cast{cval}_{name}"] = mcnemar_exact(((s.p_base > 0.5) == bool(yv)).values,
                                                            ((s.p_mod > 0.5) == bool(yv)).values)
            if d.y.nunique() == 2 and d.y.sum() >= a.min_group and (1 - d.y).sum() >= a.min_group:
                dd = d.reset_index(drop=True)
                b = paired_bootstrap(dd, lambda s: {"dAUROC": roc_auc_score(s.y, s.p_mod) - roc_auc_score(s.y, s.p_base)},
                                     a.n_boot, "cluster")
                tests[f"cast{cval}_dAUROC"] = dict(auc_base=roc_auc_score(dd.y, dd.p_base),
                                                   auc_mod=roc_auc_score(dd.y, dd.p_mod), **b["dAUROC"])
        sub = pd.DataFrame(rows)
        sub.to_csv(out / "by_cast.csv", index=False)
        res["by_cast"] = dict(rows=rows, tests=tests)
        md += ["## Stratified by cast/splint presence (rate = specificity for 'normal', sensitivity for 'abnormal')",
               sub.round(3).to_markdown(index=False), "", "Paired tests:", "```", json.dumps(tests, indent=1), "```",
               "- The key cell is cast=1 / truth=normal: it counts false positives triggered by the device itself; "
               "expect small n in MURA -> report exact CIs and treat as exploratory."]

    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "results.md").write_text("\n".join(md))
    print(f"\nWritten to {out}/ : results.md, results.json, reliability.png, by_body_part.csv"
          + (", by_cast.csv" if a.cast else ""))


if __name__ == "__main__":
    main()
