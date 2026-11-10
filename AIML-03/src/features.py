"""Step 2 - interaction/feature tables (user history, item popularity, categories).

Two interchangeable implementations that write the SAME two csv files:
  * pandas  (default, fine for ml-latest-small)   -> build_pandas()
  * Apache Spark (for ml-25m / the 'large-scale' story) -> src/spark_features.py
"""
import numpy as np
import pandas as pd
from src import config as C


def build_pandas():
    train = pd.read_csv(C.PROC / "train.csv", usecols=["user_idx", "item_idx", "rating"])
    train["pos"] = (train.rating >= C.POS_THRESHOLD).astype(int)
    item = train.groupby("item_idx").agg(item_cnt=("rating", "size"), item_avg=("rating", "mean"),
                                         item_pos=("pos", "sum")).reset_index()
    user = train.groupby("user_idx").agg(user_cnt=("rating", "size"), user_avg=("rating", "mean"),
                                         user_pos=("pos", "sum")).reset_index()
    item.to_csv(C.PROC / "item_features.csv", index=False)
    user.to_csv(C.PROC / "user_features.csv", index=False)
    print("pandas features:", len(user), "users,", len(item), "items")


def _user_genre_counts(meta, item_genre):
    """How many positives each user has per primary genre (train period only). Cached on disk."""
    cache, train = C.PROC / "user_genre_counts.npz", C.PROC / "train.csv"
    if cache.exists() and cache.stat().st_mtime > train.stat().st_mtime:
        z = np.load(cache)
        return z["counts"], z["total"]
    nu, ng = meta["n_users"], meta["n_genres"]
    df = pd.read_csv(train, usecols=["user_idx", "item_idx", "rating"])
    df = df[df.rating >= C.POS_THRESHOLD]
    flat = df.user_idx.values.astype(np.int64) * (ng + 1) + item_genre[df.item_idx.values]
    counts = np.bincount(flat, minlength=(nu + 1) * (ng + 1)).reshape(nu + 1, ng + 1).astype(np.float32)
    total = counts.sum(1)
    np.savez(cache, counts=counts, total=total)
    return counts, total


def load_features(meta):
    """Returns dense, standardized lookup arrays indexed by user_idx / item_idx (0 = unknown -> zeros)."""
    nu, ni = meta["n_users"], meta["n_items"]
    uf = pd.read_csv(C.PROC / "user_features.csv").set_index("user_idx")
    itf = pd.read_csv(C.PROC / "item_features.csv").set_index("item_idx")
    movies = pd.read_csv(C.PROC / "movies.csv").set_index("item_idx")

    def dense(df, n, cols):
        out = np.full((n + 1, len(cols)), np.nan, dtype=np.float32)
        out[df.index.values] = df[cols].values
        out[:, 0] = np.log1p(out[:, 0])
        mu, sd = np.nanmean(out[1:], 0), np.nanstd(out[1:], 0) + 1e-6
        out = (out - mu) / sd
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    user_feat = dense(uf, nu, ["user_cnt", "user_avg"])
    item_feat = dense(itf, ni, ["item_cnt", "item_avg"])
    item_genre = np.zeros(ni + 1, dtype=np.int32)
    item_genre[movies.index.values] = movies.genre_idx.values
    item_pos = np.zeros(ni + 1, dtype=np.float64)
    item_pos[itf.index.values] = itf.item_pos.values
    user_cnt = np.zeros(nu + 1, dtype=np.int64)
    user_cnt[uf.index.values] = uf.user_cnt.values
    ug_counts, ug_total = _user_genre_counts(meta, item_genre)
    return dict(user_feat=user_feat, item_feat=item_feat, item_genre=item_genre,
                item_pos=item_pos, user_cnt=user_cnt, movies=movies, ug_counts=ug_counts, ug_total=ug_total)


if __name__ == "__main__":
    build_pandas()
