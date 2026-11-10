"""Ranking metrics (pure numpy/python, no TF needed)."""
import math
import numpy as np


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & set(relevant))
    return hits / len(relevant)


def hit_at_k(ranked, relevant, k):
    return float(len(set(ranked[:k]) & set(relevant)) > 0)


def ndcg_at_k(ranked, relevant, k):
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, it in enumerate(ranked[:k]) if it in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal > 0 else 0.0


def welch_z_pvalue(a, b):
    """Two-sided p-value for difference in means (normal approximation)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return 1.0
    z = (a.mean() - b.mean()) / se
    return math.erfc(abs(z) / math.sqrt(2))
