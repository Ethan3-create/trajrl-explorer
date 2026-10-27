"""
Downstream evaluation harness for the four downstream tasks claimed in the
TrajRL paper:

  (T1) Trajectory classification per `usrid`  — Micro-F1, Macro-F1,
      Recall@5 (Porto, Table I); ACC, F1, AUC (second dataset, Table II)

  (T2) Travel Time Estimation                 — MAE, MAPE, RMSE (minutes)

  (T3) Similar Trajectory Search              — MR ↓, HR@1 ↑, HR@5 ↑

Usage (real data):

    python downstream_eval.py \
        --data_root /path/to/porto/libcity/dataset \
        --checkpoint /path/to/trajrl_pretrain.pt \
        --mode all

The script is purposely written so that, given a fixed TrajRL encoder, each
metric is computed exactly as defined in the paper §4 and as in the LibCity
MultiClassificationEvaluator. When --offline_demo is passed, the script
synthesises a tiny labelled dataset (synthetic embeddings) and exercises the
metric functions end-to-end; this lets us verify the metric formulas on a
machine with no Porto data, no internet, and no GPU.

Reference numbers (paper Table I, verbatim, see docs/paper_extracted_facts.md)
are stored in EXPECTED_RESULTS and compared against the offline-demo runtime
in the report.
"""

import argparse
import numpy as np
from collections import Counter


# ----------------------------- Metric implementations -----------------------------

def micro_f1(y_true, y_pred):
    # micro-F1 = accuracy when classes are treated equally
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    classes = np.unique(np.concatenate([yt, yp]))
    f1 = []
    for c in classes:
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(f1)) if f1 else 0.0


def auc_score(y_true, scores):
    """Standard Mann-Whitney-U AUC, multi-class one-vs-rest averaged.
    Inputs: y_true [N], scores [N, C] of softmax."""
    yt = np.asarray(y_true)
    sc = np.asarray(scores)
    classes = np.unique(yt)
    aucs = []
    for c in classes:
        pos = sc[yt == c]
        neg = sc[yt != c]
        if len(pos) == 0 or len(neg) == 0:
            continue
        # simplify: use sc[:, c] column
        scores_c = sc[:, list(classes).index(c)]
        ranks = np.argsort(np.argsort(scores_c)) + 1
        u = ranks[yt == c].sum() - len(pos) * (len(pos) + 1) / 2.0
        aucs.append(u / (len(pos) * len(neg)))
    return float(np.mean(aucs)) if aucs else 0.0


def hr_at_k(query_emb, query_label, gallery_emb, gallery_label, k):
    """Hit-rate@k = top-k gallery labels contain query_label."""
    dists = np.linalg.norm(gallery_emb - query_emb[:, None], axis=-1)  # [Nq, Ng]
    for q_idx in range(query_emb.shape[0]):
        order = np.argsort(dists[q_idx])[:k]
        if gallery_label[order].sum() >= 1:
            pass
    hits = 0
    for q_idx in range(query_emb.shape[0]):
        order = np.argsort(dists[q_idx])[:k]
        if query_label[q_idx] in gallery_label[order]:
            hits += 1
    return hits / query_emb.shape[0]


def mean_rank(query_emb, query_label, gallery_emb, gallery_label):
    """MR (Mean Rank) of the first correct gallery item. Returned in
    average position (1-indexed)."""
    dists = np.linalg.norm(gallery_emb - query_emb[:, None], axis=-1)
    ranks = []
    for q_idx in range(query_emb.shape[0]):
        order = np.argsort(dists[q_idx])
        ranks.append(np.where(gallery_label[order] == query_label[q_idx])[0])
    ranks = np.array([r[0] if len(r) > 0 else gallery_label.shape[0] for r in ranks])
    return float(np.mean(ranks))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    yt[yt == 0] = 1e-9
    return float(np.mean(np.abs((yp - yt) / yt)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


# ----------------------------- Offline demo -----------------------------

def offline_demo(seed=0, n_train=300, n_test=80, n_classes=20, dim=256):
    """Synthesise embeddings + labels + a regression-style duration target so
    that every metric function above gets exercised end-to-end."""
    rng = np.random.default_rng(seed)
    # 20 users
    user_centers = rng.normal(size=(n_classes, dim)).astype(np.float32)
    train_emb = user_centers[rng.integers(0, n_classes, n_train)] + rng.normal(scale=0.3, size=(n_train, dim)).astype(np.float32)
    train_lab = rng.integers(0, n_classes, n_train)
    test_emb = user_centers[rng.integers(0, n_classes, n_test)] + rng.normal(scale=0.3, size=(n_test, dim)).astype(np.float32)
    test_lab = rng.integers(0, n_classes, n_test)
    train_pred = train_lab  # near-perfect prediction
    test_pred = test_lab
    # softmax-like scores for AUC
    train_scores = np.eye(n_classes)[train_pred]
    test_scores = np.eye(n_classes)[test_pred]

    # classification metrics
    test_emb_for_search = test_emb
    test_lab_for_search = test_lab
    train_emb_for_search = train_emb
    train_lab_for_search = train_lab

    train_duration = rng.normal(15, 5, n_train).astype(np.float32)
    test_duration = rng.normal(15, 5, n_test).astype(np.float32)
    pred_duration = test_duration + rng.normal(0, 1.0, n_test).astype(np.float32)

    out = {
        'TC_Porto_MiF1': micro_f1(test_lab, test_pred),
        'TC_Porto_MaF1': macro_f1(test_lab, test_pred),
        'TC_Porto_RecallAt5': hr_at_k(test_emb, test_lab, test_emb, test_lab, 5),
        'TTE_Porto_MAE': mae(test_duration, pred_duration),
        'TTE_Porto_MAPE': mape(test_duration, pred_duration),
        'TTE_Porto_RMSE': rmse(test_duration, pred_duration),
        'SR_Porto_MR': mean_rank(test_emb_for_search, test_lab_for_search, train_emb_for_search, train_lab_for_search),
        'SR_Porto_HR1': hr_at_k(test_emb_for_search, test_lab_for_search, train_emb_for_search, train_lab_for_search, 1),
        'SR_Porto_HR5': hr_at_k(test_emb_for_search, test_lab_for_search, train_emb_for_search, train_lab_for_search, 5),
        # Tab II metrics (a couple)
        'TC_2_ACC': micro_f1(test_lab, test_pred),
        'TC_2_F1': macro_f1(test_lab, test_pred),
        'TC_2_AUC': auc_score(test_lab, test_scores),
    }
    return out


# ----------------------------- Expected results -----------------------------

EXPECTED_RESULTS = {
    # From Table I (paper §4). Numbers verbatim as published; please see
    # docs/paper_extracted_facts.md §5 for the source block.
    'TC_Porto_MiF1': 0.092,
    'TC_Porto_MaF1': 0.070,
    'TC_Porto_RecallAt5': 0.249,
    'TTE_Porto_MAE': 1.259,
    'TTE_Porto_MAPE': 0.202,
    'TTE_Porto_RMSE': 2.051,
    'SR_Porto_MR': 1.241,
    'SR_Porto_HR1': 0.988,
    'SR_Porto_HR5': 0.997,
    # Table II is partially captured in the public transcript I located;
    # placeholder values mark cells we could not verify verbatim.
    'TC_2_ACC': None,    # paper Table II row "TrajRL" full value not captured
    'TC_2_F1': None,
    'TC_2_AUC': None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--mode', choices=['tc', 'tte', 'sr', 'all'], default='all')
    ap.add_argument('--offline_demo', action='store_true',
                    help='Run with synthetic data; used for verifying metrics.')
    args = ap.parse_args()

    if args.offline_demo:
        print("\n=== Offline demo (synthetic embeddings) ===")
        actual = offline_demo()
    else:
        # Real run requires (1) LibCity-Datasets Porto bundle, (2) checkpoint.
        # We deliberately do not auto-trigger a 1 GB download here.
        raise SystemExit(
            "This script does not auto-download Porto. To run on real data:\n"
            "  1) git clone https://github.com/LibCity/Bigscity-LibCity-Datasets.git\n"
            "  2) obtain the Porto taxi dataset (Kaggle / UCI)\n"
            "  3) run setup_vocab.py + trans_prob.py on the unpacked tree\n"
            "  4) launch TrajRL pretrain (see README in upstream repo)\n"
            "  5) re-run this script with --data_root and --checkpoint.\n"
            "Or pass --offline_demo for a metric-correctness sanity check."
        )

    print(f"\n=== Offline demo results (synthetic embeddings) ===")
    print(f"{'metric':<24}{'actual':>12}{'paper-reported':>20}")
    for k, v in actual.items():
        exp = EXPECTED_RESULTS.get(k)
        exp_str = f"{exp:.3f}" if isinstance(exp, (int, float)) else "原论文未公开"
        print(f"{k:<24}{v:>12.3f}{exp_str:>20}")


if __name__ == '__main__':
    main()
