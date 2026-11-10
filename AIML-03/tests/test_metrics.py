import math
from src.metrics import ndcg_at_k, recall_at_k, hit_at_k, welch_z_pvalue


def test_perfect_ranking():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0
    assert recall_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0


def test_miss():
    assert ndcg_at_k([9, 8], {1}, 2) == 0.0
    assert hit_at_k([9, 8], {1}, 2) == 0.0


def test_ndcg_position_matters():
    assert math.isclose(ndcg_at_k([0, 1], {1}, 2), 1 / math.log2(3))


def test_pvalue_identical_groups():
    assert welch_z_pvalue([0, 1, 0, 1], [0, 1, 0, 1]) > 0.99
