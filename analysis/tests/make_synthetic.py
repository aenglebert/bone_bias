"""Synthetic data in the exact formats of the repo, to smoke-test the scripts (NOT real results)."""
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
rng = np.random.default_rng(0)
out = Path(__file__).parent / "synthetic"; (out/"maps").mkdir(parents=True, exist_ok=True)
parts = ["ELBOW","FINGER","FOREARM","HAND","HUMERUS","SHOULDER","WRIST"]
n = 1199
studies, y = [], []
for i in range(n):
    part = parts[i % 7]; pat = i // 2   # 2 studies per patient
    lab = int(rng.random() < 0.43)
    studies.append(f"MURA-v1.1/valid/XR_{part}/patient{11185+pat:05d}/study1_{'positive' if lab else 'negative'}/"); y.append(lab)
lab = pd.DataFrame({"s": studies, "y": y}); lab.to_csv(out/"valid_labeled_studies.csv", header=False, index=False)
y = np.array(y)
cast = (rng.random(n) < 0.10).astype(int)
# baseline: partially driven by cast; modified: less so
z = 1.6*(y-0.5)*2 + rng.normal(0,1.1,n)
sig = lambda t: 1/(1+np.exp(-t))
for tag, w in (("base", 1.8), ("mod", 0.3)):
    for f in range(5):
        p = sig(z + w*cast + rng.normal(0,0.35,n) - 0.3)
        pd.DataFrame({"0": p}).to_csv(out/f"{tag}_fold{f}_mean.csv")
pd.DataFrame({"study": studies, "cast": cast}).to_csv(out/"cast.csv", index=False)
# folds for leakage check (patient-level split, clean)
fold_df = pd.DataFrame({"s": [s.replace("valid","train").replace("patient1","patient5") for s in studies], "y": y, "pat": [s.split("/")[3].replace("patient1","patient5") for s in studies]})
pats = fold_df.pat.unique(); fmap = {p: i % 5 for i, p in enumerate(pats)}
fold_df["f"] = fold_df.pat.map(fmap)
for f in range(5):
    fold_df[fold_df.f == f][["s","y"]].to_csv(out/f"train_fold{f}.csv", header=False, index=False)
    fold_df[fold_df.f == f].head(3)[["s","y"]].to_csv(out/f"train_cast_fold{f}.csv", header=False, index=False)
# saliency maps + manifest
rows = []
H = 128
yy, xx = np.mgrid[0:H, 0:H]
blob = lambda cy, cx, s: np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*s**2))
for i in range(40):
    m = np.zeros((H,H), np.uint8); m[:, :48] = 255
    Image.fromarray(m).save(out/"maps"/f"cast{i}.png")
    sb = blob(64, 20, 10) + 0.3*blob(60, 90, 12) + 0.05*rng.random((H,H))       # baseline on cast
    sm = 0.2*blob(64, 20, 10) + blob(60, 90, 12) + 0.05*rng.random((H,H))       # modified on lesion
    np.save(out/"maps"/f"b{i}.npy", sb); np.save(out/"maps"/f"m{i}.npy", sm)
    rows.append(dict(image_id=f"img{i}", sal_base=f"maps/b{i}.npy", sal_mod=f"maps/m{i}.npy",
                     cast_mask=f"maps/cast{i}.png", lesion_boxes="0.6,0.35,0.85,0.65"))
pd.DataFrame(rows).to_csv(out/"manifest.csv", index=False)
print("synthetic data in", out)
