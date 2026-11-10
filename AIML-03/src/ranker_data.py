"""Training data + feature assembly for the ranking stage (no TensorFlow needed, so it is unit-testable).

Key idea (stacking): the retrieval model is trained on the TRAIN period. The ranker is trained on what retrieval
actually returns for each user, labelled by what that user really liked in the following VALIDATION period.
That is exactly the situation at serving time (candidates -> pick what the user will like next), it uses
HARD negatives (items the retriever thought were good but the user did not engage with), and it has no leakage
because features come from the train period only and labels come from the val period. TEST stays untouched.
"""
import numpy as np
import pandas as pd
from src import config as C

RET_SCALE = 5.0   # cosine scores are small numbers; scale them for the network


def make_inputs(users, items, feats):
    """Model inputs. num = [user_cnt, user_avg, item_cnt, item_avg, user->genre affinity, retrieval score]."""
    users, items = np.asarray(users), np.asarray(items)
    g = feats["item_genre"][items]
    aff = feats["ug_counts"][users, g] / np.maximum(feats["ug_total"][users], 1)
    ret = (feats["user_emb"][users] * feats["item_emb"][items]).sum(1) * RET_SCALE
    num = np.column_stack([feats["user_feat"][users], feats["item_feat"][items], aff, ret]).astype(np.float32)
    return {"genre": g.astype(np.int32), "num": num}


N_NUM = 6


def topk_unseen(user_emb, item_emb, users, exclude, k, chunk=512):
    """Top-k items (item_idx) per user by inner product, skipping each user's `exclude` items (zero-based arrays)."""
    E = item_emb[1:]                                   # row j  <->  item_idx j+1
    k = min(k, E.shape[0] - 1)
    out = {}
    for s in range(0, len(users), chunk):
        ub = users[s:s + chunk]
        sc = user_emb[ub] @ E.T
        for r, u in enumerate(ub):
            ex = exclude.get(u)
            if ex is not None and len(ex):
                sc[r, ex] = -np.inf
            top = np.argpartition(-sc[r], k)[:k]
            out[u] = top[np.argsort(-sc[r, top])] + 1
    return out


def build_ranker_dataset(feats, n_cand=C.N_CANDIDATES, max_users=30000, val_frac=0.15, seed=C.SEED):
    rng = np.random.default_rng(seed)
    tr = pd.read_csv(C.PROC / "train.csv", usecols=["user_idx", "item_idx"])
    seen = {u: g.item_idx.values - 1 for u, g in tr.groupby("user_idx")}
    va = pd.read_csv(C.PROC / "val.csv", usecols=["user_idx", "item_idx", "rating"])
    va = va[(va.user_idx > 0) & (va.item_idx > 0)]
    pos = {u: set(g.item_idx.values) for u, g in va[va.rating >= C.POS_THRESHOLD].groupby("user_idx")}
    users = np.array([u for u in va.user_idx.unique() if len(seen.get(u, ())) >= C.MIN_HISTORY])
    if len(users) > max_users:
        users = rng.choice(users, max_users, replace=False)
    rng.shuffle(users)
    n_val = max(1, int(len(users) * val_frac))
    parts = {"val": users[:n_val], "train": users[n_val:]}

    out = {}
    for name, us in parts.items():
        cands = topk_unseen(feats["user_emb"], feats["item_emb"], us, seen, n_cand)
        U = np.repeat(us, [len(cands[u]) for u in us])
        I = np.concatenate([cands[u] for u in us])
        y = np.array([it in pos.get(u, ()) for u, it in zip(U, I)], dtype=np.float32)
        perm = rng.permutation(len(U))
        out[name] = (make_inputs(U[perm], I[perm], feats), y[perm])
        print(f"ranker {name}: {len(us)} users, {len(y)} rows, {int(y.sum())} positives ({y.mean():.2%})")
    return out["train"], out["val"]
