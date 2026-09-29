#!/usr/bin/env python3
"""
Blinded, paired review of saliency maps (baseline vs. debiased model).

Why: judging whether the device drives the abnormal prediction is a subjective visual assessment. This script
makes it reproducible: the reader first tags device presence on the X-rays alone, then judges both models on the
same studies, in random order, with anonymised identifiers, so that an exact McNemar test can be computed.

Inputs: two folders of PNG figures (same file names in both), e.g.
    visualisations/pcama/imagenet20230820-141826_fold0-epoch=19-val_auroc=0/          (baseline)
    visualisations/pcama/cast20imagenet20230821-104820_fold0-epoch=19-val_auroc=0/    (modified, x20)
Each PNG is "<study_idx>_label<0|1>.png" = ONE STUDY: one row per radiograph (view), with the columns
X-Ray | CAM | PCAM | Occlusion and the predicted probability written on the left of each row.
The unit of analysis is the study (one annotation per study, as in the original assessment).

Workflow (three steps, ~30-40 minutes of reading in total):

  1) python saliency_review.py devices  --base DIR_BASE --mod DIR_MOD --out review
       -> review/stage1/index.html : open in a browser. For every study (X-rays only, shuffled, no model
          information): is an immobilization device (cast / splint) visible on at least one view?  keys: y / n / u(nsure), arrows.
          Click "Download CSV" and save the file as  review/stage1_devices.csv

  2) python saliency_review.py saliency --out review
       -> review/stage2/index.html : only the device-bearing studies, both models, shuffled and anonymised
          (X-ray + PCAM + displayed probability of every view). Question: does the device drive the abnormal
          prediction in this study?  Save the downloaded file as  review/stage2_saliency.csv

  3) python saliency_review.py analyze --out review [--labels valid_labeled_studies.csv
                                                     --base-preds 'imagenet2023*_fold*_mean.csv'
                                                     --mod-preds  'cast20imagenet2023*_fold*_mean.csv']
       -> review/saliency_results.md / .json : paired 2x2 table (per study), exact McNemar, counts per model, and (if the
          prediction files are given) performance with vs. without a device in the annotated studies.

The key linking anonymous ids to models never appears in the HTML pages (review/stage*/key.csv).
Dependencies: numpy, pandas, Pillow, scipy.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FNAME = re.compile(r"^(\d+)_label([01])\.png$")


# --------------------------------------------------------------------------- image layout detection
def _bands(mask, min_len):
    """Runs of True in a 1-D boolean array, as (start, end) with end exclusive, of length >= min_len."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def load_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def split_figure(path):
    """Return a list of rows; each row = dict(label=(x0,x1) or None, panels=[(x0,x1)*4], y=(y0,y1)).
    Detected from white separators between the matplotlib panels. Falls back to one row if the layout
    is not the expected 4 panels."""
    im = load_rgb(path)
    a = np.asarray(im)
    nonwhite = (a < 245).any(axis=2)
    rows = []
    for (y0, y1) in _bands(nonwhite.any(axis=1), min_len=60):      # panel rows (titles are < 60 px high)
        cols = _bands(nonwhite[y0:y1].any(axis=0), min_len=3)
        wide = [c for c in cols if c[1] - c[0] >= 100]
        narrow = [c for c in cols if c[1] - c[0] < 100 and c[0] < (wide[0][0] if wide else 0)]
        rows.append(dict(y=(y0, y1), panels=wide, label=narrow[0] if narrow else None))
    ok = rows and all(len(r["panels"]) == 4 for r in rows)
    if not ok:
        h, w = a.shape[:2]
        rows = [dict(y=(0, h), panels=[(0, w)], label=None)]
    return im, rows


def crop_row(im, row, which):
    """which: 'xray' -> X-ray panel only; 'xray+pcam' -> label + X-ray + PCAM side by side."""
    y0, y1 = row["y"]
    p = row["panels"]

    def piece(x0, x1):
        return im.crop((x0, y0, x1, y1))

    if which == "xray" or len(p) < 4:
        return piece(*p[0])
    parts = []
    if row["label"]:
        parts.append(piece(*row["label"]))
    parts += [piece(*p[0]), piece(*p[2])]
    gap = 8
    w = sum(q.width for q in parts) + gap * (len(parts) - 1)
    canvas = Image.new("RGB", (w, y1 - y0), (255, 255, 255))
    x = 0
    for q in parts:
        canvas.paste(q, (x, 0))
        x += q.width + gap
    return canvas


def tile(images, ncols, gap=12):
    """Arrange images in a grid (row-major) on a white canvas."""
    ncols = min(ncols, len(images))
    nrows = -(-len(images) // ncols)
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    canvas = Image.new("RGB", (ncols * w + (ncols - 1) * gap, nrows * h + (nrows - 1) * gap), (255, 255, 255))
    for k, im in enumerate(images):
        canvas.paste(im, ((k % ncols) * (w + gap), (k // ncols) * (h + gap)))
    return canvas


# --------------------------------------------------------------------------- HTML annotator
HTML = """<!doctype html><html><head><meta charset="utf-8"><title>%(title)s</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
#top{padding:10px 16px;background:#222;position:sticky;top:0;z-index:2}
#q{font-size:17px;margin-bottom:6px} #prog{opacity:.75;font-size:14px}
#stage{text-align:center;padding:12px} img{max-width:96vw;max-height:72vh;background:#fff}
button{font-size:16px;padding:8px 18px;margin:4px;border-radius:6px;border:0;cursor:pointer}
.y{background:#2e7d32;color:#fff}.n{background:#c62828;color:#fff}.u{background:#555;color:#fff}
.sel{outline:4px solid #ffd54f} small{opacity:.7}
</style></head><body>
<div id="top"><div id="q">%(question)s</div>
<div id="prog"></div></div>
<div id="stage"><img id="im"><div>
<button class="y" onclick="ans('1')">Yes (y)</button>
<button class="n" onclick="ans('0')">No (n)</button>
<button class="u" onclick="ans('NA')">Unsure (u)</button>
<button onclick="go(-1)">&larr; back</button>
<button onclick="dl()">Download CSV</button></div>
<small>Keys: y / n / u = answer and go to next &nbsp;|&nbsp; &larr; &rarr; = navigate &nbsp;|&nbsp; answers are kept in this browser (localStorage); download the CSV at the end (it can be re-downloaded at any time).</small></div>
<script>
const ITEMS=%(items)s, FIELD="%(field)s", KEY="review_%(stage)s";
let a=JSON.parse(localStorage.getItem(KEY)||"{}"), i=0;
const first=ITEMS.findIndex(x=>!(x.id in a)); if(first>=0) i=first;
function show(){const it=ITEMS[i];document.getElementById('im').src=it.img;
 const n=Object.keys(a).length;document.getElementById('prog').textContent=
 'item '+(i+1)+' / '+ITEMS.length+'   |   answered '+n+'   |   current: '+(a[it.id]===undefined?'-':a[it.id]);
 document.querySelectorAll('button.y,button.n,button.u').forEach(b=>b.classList.remove('sel'));
 const m={'1':'y','0':'n','NA':'u'}[a[it.id]]; if(m) document.querySelector('button.'+m).classList.add('sel');}
function ans(v){a[ITEMS[i].id]=v;localStorage.setItem(KEY,JSON.stringify(a));if(i<ITEMS.length-1)i++;show();}
function go(d){i=Math.min(ITEMS.length-1,Math.max(0,i+d));show();}
function dl(){const miss=ITEMS.filter(x=>!(x.id in a)).length;
 if(miss&&!confirm(miss+' items are not answered yet. Download anyway?'))return;
 let s='id,'+FIELD+'\\n';ITEMS.forEach(x=>{s+=x.id+','+(x.id in a?a[x.id]:'')+'\\n'});
 const b=new Blob([s],{type:'text/csv'}),l=document.createElement('a');l.href=URL.createObjectURL(b);
 l.download='%(outname)s';l.click();}
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();
 if(k==='y'||k==='1')ans('1');else if(k==='n'||k==='0')ans('0');else if(k==='u')ans('NA');
 else if(e.key==='ArrowRight')go(1);else if(e.key==='ArrowLeft')go(-1);});
show();
</script></body></html>"""


def write_html(folder, items, question, field, stage, outname, title):
    (folder / "index.html").write_text(HTML % dict(
        title=title, question=question, items=json.dumps(items), field=field, stage=stage, outname=outname))


# --------------------------------------------------------------------------- stage 1: devices
def cmd_devices(a):
    out = Path(a.out) / "stage1"
    (out / "img").mkdir(parents=True, exist_ok=True)
    base, mod = Path(a.base), Path(a.mod)
    fb = {p.name: p for p in base.glob("*.png") if FNAME.match(p.name)}
    fm = {p.name: p for p in mod.glob("*.png") if FNAME.match(p.name)}
    common = sorted(set(fb) & set(fm), key=lambda s: int(s.split("_")[0]))
    if set(fb) ^ set(fm):
        print(f"WARNING: {len(set(fb) ^ set(fm))} matching file(s) present in only one folder are ignored: "
              f"{sorted(set(fb) ^ set(fm))[:5]}", file=sys.stderr)
    order = list(range(len(common)))
    random.Random(a.seed).shuffle(order)
    items, key = [], []
    for n, k in enumerate(order, 1):
        name = common[k]
        idx, label = FNAME.match(name).groups()
        imb, rb = split_figure(fb[name])
        imm, rm = split_figure(fm[name])
        if len(rb) != len(rm):
            print(f"WARNING: {name}: {len(rb)} views in base, {len(rm)} in modified", file=sys.stderr)
        did = f"d{n:04d}"
        tile([crop_row(imb, r, "xray") for r in rb], ncols=4).save(out / "img" / f"{did}.png")
        items.append(dict(id=did, img=f"img/{did}.png"))
        key.append(dict(id=did, idx=int(idx), label=int(label), file=name, n_views_base=len(rb),
                        n_views_mod=len(rm)))
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    write_html(out, items, "Is an immobilization device (cast or splint) visible on at least one view of this study?",
               "device", "stage1", "stage1_devices.csv", "Stage 1 - devices")
    (Path(a.out) / "config.json").write_text(json.dumps(dict(base=str(base), mod=str(mod), seed=a.seed)))
    print(f"{len(items)} studies -> open {out / 'index.html'}")
    print("Save the downloaded file as", Path(a.out) / "stage1_devices.csv")


# --------------------------------------------------------------------------- stage 2: saliency
def cmd_saliency(a):
    root = Path(a.out)
    cfg = json.loads((root / "config.json").read_text())
    key1 = pd.read_csv(root / "stage1" / "key.csv")
    dev = pd.read_csv(a.devices or root / "stage1_devices.csv", dtype=str, keep_default_na=False)
    k = key1.merge(dev, on="id")
    n_na = int((~k.device.isin(["0", "1"])).sum())
    k = k[k.device == "1"]
    print(f"{len(k)} device-bearing studies ({n_na} unanswered/unsure ignored)")
    out = root / "stage2"
    (out / "img").mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"] + 1)
    recs = [dict(idx=int(r.idx), label=int(r.label), file=r.file, model=m, folder=f, device_id=r.id)
            for _, r in k.iterrows() for m, f in (("base", cfg["base"]), ("mod", cfg["mod"]))]
    rng.shuffle(recs)
    items, key = [], []
    for n, r in enumerate(recs, 1):
        sid = f"s{n:04d}"
        im, rows = split_figure(Path(r["folder"]) / r["file"])
        tile([crop_row(im, row, "xray+pcam") for row in rows], ncols=2).save(out / "img" / f"{sid}.png")
        items.append(dict(id=sid, img=f"img/{sid}.png"))
        key.append(dict(id=sid, model=r["model"], idx=r["idx"], label=r["label"], file=r["file"],
                        device_id=r["device_id"]))
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    q = ("Each view: predicted probability of 'abnormal' (left), radiograph (middle), PolyCAM saliency (right). "
         "In this study, does the immobilization device drive the abnormal prediction "
         "(a view with probability &ge; 0.5 in which the device is a main highlighted area)? "
         "If no view is predicted abnormal, answer No.")
    write_html(out, items, q, "drives", "stage2", "stage2_saliency.csv", "Stage 2 - saliency")
    print(f"{len(items)} maps ({len(items) // 2} paired studies) -> open {out / 'index.html'}")
    print("Save the downloaded file as", root / "stage2_saliency.csv")


# --------------------------------------------------------------------------- analysis
def mcnemar_exact(b, c):
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue)


def cp_ci(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def cmd_analyze(a):
    root = Path(a.out)
    key = pd.read_csv(root / "stage2" / "key.csv")
    ans = pd.read_csv(a.saliency or root / "stage2_saliency.csv", dtype=str, keep_default_na=False)
    d = key.merge(ans, on="id")
    d["drives"] = pd.to_numeric(d["drives"], errors="coerce")
    piv = d.pivot_table(index="idx", columns="model", values="drives", aggfunc="first")
    lost = int(piv.isna().any(axis=1).sum())
    piv = piv.dropna()
    x, y = piv["base"].astype(int), piv["mod"].astype(int)
    tab = pd.crosstab(x, y).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    b, c = int(tab.loc[1, 0]), int(tab.loc[0, 1])
    res = dict(n_studies=int(len(piv)), n_dropped_missing=lost,
               table=dict(both=int(tab.loc[1, 1]), base_only=b, mod_only=c, neither=int(tab.loc[0, 0])),
               device_drives_base=int(x.sum()), device_drives_mod=int(y.sum()),
               mcnemar_exact_p=mcnemar_exact(b, c))
    lines = ["## Blinded paired saliency review (device drives the abnormal prediction; unit = study)",
             f"- device-bearing studies judged for both models: {res['n_studies']} "
             f"({lost} dropped for missing/unsure answers)",
             f"- baseline: {res['device_drives_base']}; modified: {res['device_drives_mod']}",
             f"- paired table: both={res['table']['both']}, baseline only={b}, modified only={c}, "
             f"neither={res['table']['neither']}",
             f"- exact McNemar p = {res['mcnemar_exact_p']:.4g}", ""]

    # optional: performance with / without device among the reviewed studies
    if a.labels and a.base_preds and a.mod_preds:
        import glob
        lab = pd.read_csv(a.labels, header=None, names=["study", "y"])
        key1 = pd.read_csv(root / "stage1" / "key.csv")
        dev = pd.read_csv(a.devices or root / "stage1_devices.csv", dtype=str, keep_default_na=False)
        k1 = key1.merge(dev, on="id")
        k1["device"] = pd.to_numeric(k1["device"], errors="coerce")
        per_study = k1.set_index("idx").device                  # 1 = device on >= 1 view; NaN = unsure
        per_study = per_study[per_study.isin([0, 1])]
        chk = k1.set_index("idx").label
        bad = [i for i in per_study.index if int(lab.y.iloc[i]) != int(chk[i])]
        if bad:
            print(f"WARNING: filename label != labels file for study indices {bad[:10]} - "
                  f"check that both refer to the same ordering", file=sys.stderr)

        def ens(pat):
            fs = sorted(glob.glob(pat))
            return np.mean([pd.read_csv(f, index_col=0).iloc[:, 0].values for f in fs], axis=0), len(fs)
        pb, nb = ens(a.base_preds)
        pm, nm = ens(a.mod_preds)
        rows = []
        for has in (1, 0):
            for truth, name in ((0, "Normal (specificity)"), (1, "Abnormal (sensitivity)")):
                idx = [i for i in per_study.index if per_study[i] == has and int(lab.y.iloc[i]) == truth]
                if not idx:
                    continue
                cb = [(pb[i] >= .5) == truth for i in idx]
                cm = [(pm[i] >= .5) == truth for i in idx]
                bo = sum(1 for u, v in zip(cb, cm) if u and not v)
                mo = sum(1 for u, v in zip(cb, cm) if v and not u)
                rows.append(dict(device="present" if has else "absent", truth=name, n=len(idx),
                                 base=f"{sum(cb)}/{len(idx)} ({sum(cb) / len(idx):.2f}; "
                                      f"{cp_ci(sum(cb), len(idx))[0]:.2f}-{cp_ci(sum(cb), len(idx))[1]:.2f})",
                                 mod=f"{sum(cm)}/{len(idx)} ({sum(cm) / len(idx):.2f}; "
                                     f"{cp_ci(sum(cm), len(idx))[0]:.2f}-{cp_ci(sum(cm), len(idx))[1]:.2f})",
                                 p=round(mcnemar_exact(bo, mo), 3)))
        tbl = pd.DataFrame(rows)
        res["device_subgroup"] = rows
        lines += [f"## Performance in the reviewed studies with / without a device (ensembles of {nb} / {nm} "
                  f"fold models, threshold 0.5, exact CIs)", tbl.to_markdown(index=False), ""]
    (root / "saliency_results.md").write_text("\n".join(lines))
    (root / "saliency_results.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p1 = sp.add_parser("devices")
    p1.add_argument("--base", required=True)
    p1.add_argument("--mod", required=True)
    p1.add_argument("--out", default="review")
    p1.add_argument("--seed", type=int, default=0)
    p2 = sp.add_parser("saliency")
    p2.add_argument("--out", default="review")
    p2.add_argument("--devices", help="default: <out>/stage1_devices.csv")
    p3 = sp.add_parser("analyze")
    p3.add_argument("--out", default="review")
    p3.add_argument("--saliency", help="default: <out>/stage2_saliency.csv")
    p3.add_argument("--devices", help="default: <out>/stage1_devices.csv")
    p3.add_argument("--labels")
    p3.add_argument("--base-preds")
    p3.add_argument("--mod-preds")
    a = ap.parse_args()
    {"devices": cmd_devices, "saliency": cmd_saliency, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
