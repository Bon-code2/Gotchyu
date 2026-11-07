"""Train the Gotchyu classifier and save gotchyu_model.joblib.

  pip install torch transformers datasets scikit-learn joblib numpy
  python train_detector.py --questions 1500

Optional extra data (put .txt files in a folder, one text per file):
  --human-dir my_human_texts   add formal academic and non-native English human writing!
  --ai-dir    my_ai_texts      add output from current models (HC3 is 2022-era ChatGPT)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyzer import FEATURE_NAMES, Detector, GPT2Scorer


def load_hc3(n_questions: int, seed: int, min_words: int, max_words: int):
    from datasets import load_dataset
    import sys
    try:
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
    except Exception as e:
        sys.exit(f"Could not load HC3 ({e}).\nTry: pip install 'datasets<4'  (newer versions dropped dataset scripts).")
    ds = ds.shuffle(seed=seed)
    ok = lambda t: min_words <= len(t.split()) <= max_words
    texts, labels, groups = [], [], []
    for gi, row in enumerate(ds):
        human = next((t for t in row["human_answers"] if ok(t)), None)

        # Only append the human baseline text (label 0)
        if human:
            texts.append(human)
            labels.append(0)
            groups.append(gi)

        if len(texts) >= n_questions:
            break

    return texts, labels, groups


def load_dir(path: str, label: int, group_start: int, min_words: int):
    import os
    import glob
    texts, labels, groups = [], [], []
    # Changed "*.txt" to "*" to catch your files that don't have extensions
    for i, f in enumerate(sorted(glob.glob(os.path.join(path, "*")))):
        if os.path.isfile(f):  # make sure it's a file, not a subfolder
            t = open(f, encoding="utf-8", errors="ignore").read()
            if len(t.split()) >= min_words:
                texts.append(t)
                labels.append(label)
                groups.append(group_start + i)
    return texts, labels, groups


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def fit_and_evaluate(X, y, groups, seed: int = 42):
    """Hold out whole questions (groups), report honest metrics, then refit on everything."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y, groups = np.asarray(y), np.asarray(groups)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
    folds = lambda idx: list(GroupKFold(5).split(X[idx], y[idx], groups[idx]))
    cal = CalibratedClassifierCV(make_model(), method="sigmoid", cv=folds(tr)).fit(X[tr], y[tr])
    p = cal.predict_proba(X[te])[:, 1]
    fpr, tpr, _ = roc_curve(y[te], p)
    metrics = {
        "auc": float(roc_auc_score(y[te], p)),
        "tpr_at_5pct_fpr": float(tpr[fpr <= 0.05].max()),
        "tpr_at_1pct_fpr": float(tpr[fpr <= 0.01].max()),
        "human_flagged_at_50": float(((p >= 0.5) & (y[te] == 0)).sum() / max((y[te] == 0).sum(), 1)),
        "ai_caught_at_50": float(((p >= 0.5) & (y[te] == 1)).sum() / max((y[te] == 1).sum(), 1)),
        "n_train": int(len(tr)), "n_test": int(len(te)),
    }
    coefs = make_model().fit(X[tr], y[tr])[-1].coef_[0]
    final = CalibratedClassifierCV(make_model(), method="sigmoid", cv=folds(np.arange(len(y)))).fit(X, y)
    return final, metrics, coefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=1500)
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--max-words", type=int, default=700)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--human-dir")
    ap.add_argument("--ai-dir")
    ap.add_argument("--cache", default="features_cache.npz", help="feature cache so retraining is fast")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and recompute features")
    ap.add_argument("--out", default="gotchyu_model.joblib")
    a = ap.parse_args()

    if os.path.exists(a.cache) and not a.refresh:
        d = np.load(a.cache)
        X, y, groups = d["X"], d["y"], d["groups"]
        print(f"Loaded {len(y)} cached feature rows from {a.cache}")
    else:
        texts, labels, groups = load_hc3(a.questions, a.seed, a.min_words, a.max_words)
        for path, label, start in ((a.human_dir, 0, 10**7), (a.ai_dir, 1, 2 * 10**7)):
            if path:
                t, l, g = load_dir(path, label, start, a.min_words)
                texts, labels, groups = texts + t, labels + l, groups + g
        print(f"Scoring {len(texts)} texts with GPT-2 (slow on CPU; results are cached)...")
        det = Detector(model_path="__none__", lm=GPT2Scorer())
        rows, keep = [], []
        for i, t in enumerate(texts):
            v = det.featurize(t)
            if v is not None:
                rows.append(v)
                keep.append(i)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(texts)}")
        X = np.array(rows)
        y, groups = np.array(labels)[keep], np.array(groups)[keep]
        np.savez(a.cache, X=X, y=y, groups=groups)

    model, m, coefs = fit_and_evaluate(X, y, groups, a.seed)
    print("\nHeld-out results (whole questions held out, never seen in training):")
    print(f"  AUC {m['auc']:.3f} | human text flagged at 50%: {m['human_flagged_at_50']:.1%} | "
          f"AI caught at 50%: {m['ai_caught_at_50']:.1%}")
    print(f"  AI caught at 5% false positives: {m['tpr_at_5pct_fpr']:.1%} | at 1%: {m['tpr_at_1pct_fpr']:.1%}")
    if m["auc"] > 0.98:
        print("  Warning: very high AUC on HC3 often means shortcuts. Test on your own texts before trusting it.")
    print("\nFeature weights (scaled):")
    for name, c in sorted(zip(FEATURE_NAMES, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:20s} {c:+.2f}")
    for k, cls in ((0, "human"), (1, "AI")):
        print(f"Median token NLL, {cls}: {np.median(X[y == k][:, 0]):.2f}  (compare with NLL_LOW/NLL_HIGH in ml_analyzer.py)")
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "gpt2_id": "gpt2", "metrics": m,
                 "trained_on": "HC3 (paired, length-filtered)" + (" + custom folders" if a.human_dir or a.ai_dir else "")}, a.out)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()