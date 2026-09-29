# Analysis scripts

Additional analyses used in the paper. They only need the outputs of `eval.py` (study-level `*_mean.csv`
files), the MURA validation labels (`valid_labeled_studies.csv`, used as the test set) and, for the saliency
review, the PNG figures written by `resnetexpl.py`. Extra requirements are in `requirements.txt`
(numpy, scipy, scikit-learn, tabulate).

- `check_patient_leakage.py` – checks that the cross-validation folds (and the cast crops) are
  patient-disjoint and that no training patient appears in the test set.
- `compare_models.py` – paired comparison of two models (baseline vs. cast-augmented): paired DeLong test,
  paired patient-cluster bootstrap of the differences, exact McNemar test, non-inferiority check,
  calibration (Brier, ECE, slope, intercept, reliability diagram), per-region and per-class comparison.

  ```
  python analysis/compare_models.py --labels valid_labeled_studies.csv \
      --base 'imagenet2023*_fold*_mean.csv' --mod 'cast20imagenet2023*_fold*_mean.csv' --out compare_results
  ```
- `saliency_review.py` – blinded, paired review of saliency maps (device presence on X-rays alone, then
  "does the device drive the abnormal prediction?" for both models, shuffled and anonymised), with exact McNemar
  test. See the docstring for the three steps (`devices`, `saliency`, `analyze`).
- `tests/make_synthetic.py` – generates synthetic data in the formats of the repo to smoke-test the scripts
  (no real results).
