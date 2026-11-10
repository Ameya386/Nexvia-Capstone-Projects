"""Step 7b - simple online-style A/B test (replayed offline on the future test period).
Users are bucketed 50/50 by a hash of user id: bucket A is served Wide&Deep, bucket B is served DCN.
A recommended item counts as a *click* if the user really interacted with it positively in the test period.
Reports click-through (clicks / impressions), NDCG@10 and a significance test.

    python -m src.ab_test
"""
import hashlib, json
import numpy as np
from src import config as C
from src.metrics import ndcg_at_k, welch_z_pvalue
from src.recommend import Recommender
from src.evaluate import test_truth, eval_users

K = 10


def bucket(user_id):
    return "A" if int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 2 == 0 else "B"


def main():
    rec = Recommender(seen_splits=("train", "val"))
    truth = test_truth(rec)
    users = eval_users(rec, truth, n=6000)
    arms = {"A": ("wide_deep", []), "B": ("dcn", [])}
    for u in users:
        b = bucket(u)
        kind = arms[b][0]
        cands, _ = rec.retrieve(u)
        probs = rec.score_pairs(kind, np.full(len(cands), u), cands)
        top = cands[np.argsort(-probs)[:K]].tolist()
        clicks = len(set(top) & truth[u])
        arms[b][1].append((clicks / K, ndcg_at_k(top, truth[u], K), clicks))

    res = {}
    for b, (kind, rows) in arms.items():
        a = np.array(rows)
        res[b] = {"model": kind, "users": len(rows), "impressions": len(rows) * K, "clicks": int(a[:, 2].sum()),
                  "CTR": float(a[:, 0].mean()), "NDCG@10": float(a[:, 1].mean())}
    res["p_value_CTR"] = welch_z_pvalue([r[0] for r in arms["A"][1]], [r[0] for r in arms["B"][1]])
    print(json.dumps(res, indent=2))
    (C.RESULTS / "ab_test.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
