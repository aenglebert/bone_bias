#!/usr/bin/env python3
"""
Checks that the cross-validation folds are patient-disjoint.

Reads the fold CSVs used for training (train_fold{0..4}.csv, train_cast_fold{0..4}.csv) and the test CSV
(valid_labeled_studies.csv). MURA study paths look like
    MURA-v1.1/train/XR_WRIST/patient00012/study1_positive/
so the patient can be recovered from the path.

Reports
  * patients present in more than one CV fold          (leakage between training and validation folds)
  * patients present in both train folds and the test  (should be 0: MURA's official split is patient-level)
  * cast crops whose source study lives in another fold than the fold they were assigned to
Exit code 1 if any overlap is found, so it can be used in a script.
"""
import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

PAT = re.compile(r"(patient\d+)")


def patients(df):
    return set(PAT.search(str(p)).group(1) for p in df[0] if PAT.search(str(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="directory holding train_fold*.csv (as in mura.py: data_dir)")
    ap.add_argument("--test-csv", default=None, help="valid_labeled_studies.csv (optional; the official MURA split is patient-disjoint)")
    ap.add_argument("--n-folds", type=int, default=5)
    a = ap.parse_args()
    d = Path(a.data_dir)

    folds, casts = {}, {}
    for i in range(a.n_folds):
        folds[i] = pd.read_csv(d / f"train_fold{i}.csv", header=None)
        cf = d / f"train_cast_fold{i}.csv"
        casts[i] = pd.read_csv(cf, header=None) if cf.exists() else None
    test = pd.read_csv(a.test_csv, header=None) if a.test_csv else None

    bad = False
    print("Fold sizes (studies / patients):")
    for i, f in folds.items():
        print(f"  fold {i}: {len(f)} / {len(patients(f))}")

    # 1. patient overlap between folds
    for i, j in combinations(folds, 2):
        ov = patients(folds[i]) & patients(folds[j])
        if ov:
            bad = True
        print(f"folds {i}-{j}: {len(ov)} shared patients" + (f"  e.g. {sorted(ov)[:3]}" if ov else ""))

    # 2. train vs test
    tr = set().union(*[patients(f) for f in folds.values()])
    if test is not None:
        ov = tr & patients(test)
        bad |= bool(ov)
        print(f"train folds vs test: {len(ov)} shared patients")
    else:
        print(f"train folds: {len(tr)} patients, {sum(len(f) for f in folds.values())} studies (test csv not given)")

    # 3. study duplicated across folds (weaker check, should never happen)
    allstudies = pd.concat([f.assign(fold=i) for i, f in folds.items()])
    dup = allstudies[allstudies.duplicated(0, keep=False)]
    print(f"studies present in >1 fold: {dup[0].nunique()}")
    bad |= len(dup) > 0

    # 4. cast crops vs their fold
    if all(c is not None for c in casts.values()):
        owner = {p: i for i, f in folds.items() for p in patients(f)}
        mism = 0
        for i, c in casts.items():
            for p in patients(c):
                if owner.get(p, i) != i:
                    mism += 1
        bad |= mism > 0
        print(f"cast-crop patients assigned to a fold different from their own studies' fold: {mism}")
    print("\nRESULT:", "LEAKAGE FOUND -> re-split at patient level (GroupKFold) and retrain, or disclose as a limitation"
          if bad else "no patient-level leakage detected")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
