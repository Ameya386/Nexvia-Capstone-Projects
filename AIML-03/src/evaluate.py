"""Step 7a - offline evaluation on the held-out FUTURE (test) period.
Compares: popularity baseline | retrieval only | retrieval + Wide&Deep | retrieval + DCN
Metrics: NDCG@10, Recall@10, Recall@200 (candidate recall), HitRate@10.

    python -m src.evaluate            # writes results/metrics.json, metrics.md, metrics.png
"""
import json
import numpy as np
import pandas as pd
from src import config as C
from src.metrics import ndcg_at_k, recall_at_k, hit_at_k
from src.recommend import Recommender

N_EVAL_USERS = 3000


def test_truth(rec):
    te = pd.read_csv(C.PROC / "test.csv", usecols=["user_idx", "item_idx", "rating"])
    te = te[(te.user_idx > 0) & (te.item_idx > 0) & (te.rating >= C.POS_THRESHOLD)]
    return {u: set(g.item_idx.values) for u, g in te.groupby("user_idx")}


def eval_users(rec, truth, n=N_EVAL_USERS):
    users = np.array([u for u in truth if len(rec.seen.get(u, ())) >= C.MIN_HISTORY])
    rng = np.random.default_rng(C.SEED)
    return rng.choice(users, min(n, len(users)), replace=False)


def summarize(name, ranked_lists, users, truth, cand_lists=None):
    row = {"model": name,
           "NDCG@10": np.mean([ndcg_at_k(r, truth[u], 10) for r, u in zip(ranked_lists, users)]),
           "Recall@10": np.mean([recall_at_k(r, truth[u], 10) for r, u in zip(ranked_lists, users)]),
           "HitRate@10": np.mean([hit_at_k(r, truth[u], 10) for r, u in zip(ranked_lists, users)])}
    if cand_lists is not None:
        row["Recall@200"] = np.mean([recall_at_k(c, truth[u], 200) for c, u in zip(cand_lists, users)])
    return row


def main():
    C.RESULTS.mkdir(exist_ok=True, parents=True)
    rec = Recommender(seen_splits=("train", "val"))      # test period is the unseen future
    truth = test_truth(rec)
    users = eval_users(rec, truth)
    print("evaluating on", len(users), "users")

    rows = []
    pop = [rec.popular(rec.seen.get(u, set()), 200) for u in users]
    rows.append(summarize("Popularity baseline", pop, users, truth, pop))

    cands = [rec.retrieve(u)[0].tolist() for u in users]
    rows.append(summarize("Two-tower retrieval only", cands, users, truth, cands))

    flat_u = np.concatenate([[u] * len(c) for u, c in zip(users, cands)])
    flat_i = np.concatenate(cands)
    cuts = np.cumsum([len(c) for c in cands])[:-1]
    for kind, label in (("wide_deep", "A: Retrieval + Wide&Deep"), ("dcn", "B: Retrieval + DCN")):
        probs = np.split(rec.score_pairs(kind, flat_u, flat_i), cuts)
        ranked = [np.array(c)[np.argsort(-p)].tolist() for c, p in zip(cands, probs)]
        rows.append(summarize(label, ranked, users, truth, cands))

    df = pd.DataFrame(rows).set_index("model").round(4)
    print(df)
    df.to_json(C.RESULTS / "metrics.json", indent=2)
    (C.RESULTS / "metrics.md").write_text(df.to_markdown() if _has_tabulate() else df.to_string())

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = df[["NDCG@10", "Recall@10", "HitRate@10"]].plot.bar(figsize=(9, 4.5), rot=15)
    ax.set_title(f"Offline evaluation on time-held-out test set ({len(users)} users)")
    plt.tight_layout(); plt.savefig(C.RESULTS / "metrics.png", dpi=150)


def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
