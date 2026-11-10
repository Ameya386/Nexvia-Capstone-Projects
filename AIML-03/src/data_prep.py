"""Step 1 - get MovieLens, clean it, and split BY TIME (not randomly).

    python -m src.data_prep --dataset small          # downloads ml-latest-small
    python -m src.data_prep --dataset 25m            # downloads ml-25m (~250MB)
    python -m src.data_prep --synthetic              # offline fake data, for smoke tests
"""
import argparse, json, urllib.request, zipfile
import numpy as np
import pandas as pd
from src import config as C

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western", "Film-Noir"]


def download(dataset):
    folder, url = C.DATASETS[dataset]
    target = C.RAW / folder
    if target.exists():
        return target
    C.RAW.mkdir(parents=True, exist_ok=True)
    zpath = C.RAW / f"{folder}.zip"
    print(f"Downloading {url} ...")
    import ssl
    ctx = ssl._create_unverified_context()   # grouplens.org's certificate chain fails verification on Colab
    zpath.write_bytes(urllib.request.urlopen(url, context=ctx).read())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(C.RAW)
    return target


def synthetic(n_users=2000, n_items=1500, n_inter=150_000, seed=C.SEED):
    """Fake ratings with real latent structure so models can actually learn."""
    rng = np.random.default_rng(seed)
    d = 8
    uf, itf = rng.normal(size=(n_users, d)), rng.normal(size=(n_items, d))
    pop = rng.zipf(1.6, n_items).astype(float); pop /= pop.sum()
    users = rng.integers(0, n_users, n_inter)
    items = rng.choice(n_items, n_inter, p=pop)
    score = (uf[users] * itf[items]).sum(1) + rng.normal(0, 1, n_inter)
    rating = np.clip(np.round((3 + score) * 2) / 2, 0.5, 5.0)
    ts = rng.integers(1_400_000_000, 1_600_000_000, n_inter)
    ratings = pd.DataFrame({"userId": users + 1, "movieId": items + 1, "rating": rating, "timestamp": ts})
    ratings = ratings.drop_duplicates(["userId", "movieId"])
    g1 = np.argmax(itf[:, :len(GENRES)] if d >= len(GENRES) else np.pad(itf, ((0, 0), (0, len(GENRES) - d))), 1)
    g2 = rng.integers(0, len(GENRES), n_items)
    movies = pd.DataFrame({"movieId": np.arange(n_items) + 1,
                           "title": [f"Synthetic Movie {i+1} ({1980 + i % 40})" for i in range(n_items)],
                           "genres": [f"{GENRES[a]}|{GENRES[b]}" for a, b in zip(g1, g2)]})
    return ratings, movies


def prepare(dataset="small", use_synthetic=False):
    C.ensure_dirs()
    if use_synthetic:
        ratings, movies = synthetic()
    else:
        d = download(dataset)
        ratings = pd.read_csv(d / "ratings.csv")
        movies = pd.read_csv(d / "movies.csv")

    ratings = ratings.drop_duplicates(["userId", "movieId"]).dropna()
    ratings = ratings[ratings.movieId.isin(movies.movieId)].sort_values("timestamp").reset_index(drop=True)

    # ---- time-based split (per user, chronological)
    t1, t2 = ratings.timestamp.quantile([C.TRAIN_Q, C.VAL_Q])
    # Per-user chronological split: every user's OLDEST 80% -> train, next 10% -> val, NEWEST 10% -> test.
    # (A single global cutoff leaves almost no users present in both train and test on MovieLens.)
    pct = ratings.groupby("userId").cumcount() / ratings.groupby("userId")["timestamp"].transform("size")
    split = np.where(pct < C.TRAIN_Q, "train", np.where(pct < C.VAL_Q, "val", "test"))
    ratings["split"] = split

    train = ratings[ratings.split == "train"]
    user_map = {int(u): i + 1 for i, u in enumerate(sorted(train.userId.unique()))}   # 0 = unknown
    item_map = {int(m): i + 1 for i, m in enumerate(sorted(train.movieId.unique()))}
    ratings["user_idx"] = ratings.userId.map(user_map).fillna(0).astype(int)
    ratings["item_idx"] = ratings.movieId.map(item_map).fillna(0).astype(int)

    movies = movies[movies.movieId.isin(item_map)].copy()
    movies["item_idx"] = movies.movieId.map(item_map)
    movies["genres"] = movies.genres.replace("(no genres listed)", "Unknown")
    movies["primary_genre"] = movies.genres.str.split("|").str[0]
    gvocab = {g: i + 1 for i, g in enumerate(sorted(movies.primary_genre.unique()))}
    movies["genre_idx"] = movies.primary_genre.map(gvocab).astype(int)
    movies = movies.sort_values("item_idx")

    for name in ("train", "val", "test"):
        ratings[ratings.split == name].drop(columns="split").to_csv(C.PROC / f"{name}.csv", index=False)
    movies.to_csv(C.PROC / "movies.csv", index=False)
    meta = {"n_users": len(user_map), "n_items": len(item_map), "n_genres": len(gvocab),
            "genre_vocab": gvocab, "user_map": {str(k): v for k, v in user_map.items()},
            "item_map": {str(k): v for k, v in item_map.items()},
            "split_times": [int(t1), int(t2)], "dataset": "synthetic" if use_synthetic else dataset}
    (C.PROC / "meta.json").write_text(json.dumps(meta))
    print({k: v for k, v in meta.items() if k in ("n_users", "n_items", "n_genres", "dataset")})
    print("rows  train/val/test:", [int((split == s).sum()) for s in ("train", "val", "test")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="small", choices=list(C.DATASETS))
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    prepare(a.dataset, a.synthetic)
