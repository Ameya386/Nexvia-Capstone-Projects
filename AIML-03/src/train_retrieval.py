"""Step 3 - two-tower retrieval model in TensorFlow Recommenders.
User tower: user id -> embedding.  Item tower: item id + genre -> embedding.
Trained with in-batch sampled softmax (with popularity-bias correction). Exports L2-normalised embeddings.

    python -m src.train_retrieval
"""
import json
import numpy as np
import pandas as pd
from src import config as C
import tensorflow as tf
import tensorflow_recommenders as tfrs
from src.features import load_features

L = tf.keras.layers


def build_towers(n_users, n_items, n_genres, dim=C.EMB_DIM):
    u_in = tf.keras.Input(shape=(), dtype=tf.int32, name="user")
    u = L.Embedding(n_users + 1, dim)(u_in)
    u = L.Dense(dim)(L.Dense(dim, activation="relu")(u))
    user_tower = tf.keras.Model(u_in, u, name="user_tower")

    i_in = tf.keras.Input(shape=(), dtype=tf.int32, name="item")
    g_in = tf.keras.Input(shape=(), dtype=tf.int32, name="genre")
    i = L.Concatenate()([L.Embedding(n_items + 1, dim)(i_in), L.Embedding(n_genres + 1, 16)(g_in)])
    i = L.Dense(dim)(L.Dense(dim, activation="relu")(i))
    item_tower = tf.keras.Model([i_in, g_in], i, name="item_tower")
    return user_tower, item_tower


class TwoTower(tfrs.Model):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower, self.item_tower = user_tower, item_tower
        self.task = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            temperature=C.TEMPERATURE, remove_accidental_hits=True)

    def compute_loss(self, features, training=False):
        q = tf.math.l2_normalize(self.user_tower(features["user"]), axis=1)
        c = tf.math.l2_normalize(self.item_tower([features["item"], features["genre"]]), axis=1)
        return self.task(q, c, candidate_sampling_probability=features["prob"],
                         candidate_ids=features["item"])


def main():
    tf.random.set_seed(C.SEED)
    meta = json.loads((C.PROC / "meta.json").read_text())
    feats = load_features(meta)
    train = pd.read_csv(C.PROC / "train.csv", usecols=["user_idx", "item_idx", "rating"])
    pos = train[train.rating >= C.POS_THRESHOLD]
    prob = feats["item_pos"] / feats["item_pos"].sum()             # how often an item shows up as a positive
    ds = tf.data.Dataset.from_tensor_slices({
        "user": pos.user_idx.values.astype(np.int32),
        "item": pos.item_idx.values.astype(np.int32),
        "genre": feats["item_genre"][pos.item_idx.values],
        "prob": np.clip(prob[pos.item_idx.values], 1e-9, 1).astype(np.float32),
    }).shuffle(200_000, seed=C.SEED).batch(512 if len(pos) < 500_000 else C.RETRIEVAL_BATCH).prefetch(tf.data.AUTOTUNE)

    ut, it = build_towers(meta["n_users"], meta["n_items"], meta["n_genres"])
    model = TwoTower(ut, it)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    hist = model.fit(ds, epochs=(30 if len(pos) < 500_000 else C.RETRIEVAL_EPOCHS))

    C.ART.mkdir(exist_ok=True, parents=True)
    norm = lambda x: (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
    uemb = norm(ut.predict(np.arange(meta["n_users"] + 1, dtype=np.int32), batch_size=8192, verbose=0))
    iemb = norm(it.predict([np.arange(meta["n_items"] + 1, dtype=np.int32), feats["item_genre"]],
                           batch_size=8192, verbose=0))
    np.save(C.ART / "user_emb.npy", uemb)
    np.save(C.ART / "item_emb.npy", iemb)
    json.dump({"loss": [float(x) for x in hist.history["loss"]]}, open(C.ART / "retrieval_history.json", "w"))
    print("saved embeddings", uemb.shape, iemb.shape)


if __name__ == "__main__":
    main()
