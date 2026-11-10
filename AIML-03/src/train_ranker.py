"""Step 5 - deep ranking models that re-score retrieved candidates.
Two variants (used later for the A/B test):
    A = Wide & Deep      B = DCN (Deep & Cross Network, from TFRS)
Training data = the retrieval stage's own candidates per user (hard negatives), labelled by what the user liked in the
validation period - see src/ranker_data.py.

    python -m src.train_ranker
"""
import json
import numpy as np
import pandas as pd
from src import config as C
import tensorflow as tf
import tensorflow_recommenders as tfrs
from src.features import load_features
from src.ranker_data import build_ranker_dataset, N_NUM

L = tf.keras.layers
KINDS = {"wide_deep": "A", "dcn": "B"}


def build_ranker(kind, n_users, n_items, n_genres):
    # NOTE: no user/item ID embeddings on purpose - with only a few thousand labelled positives they overfit
    # (n_users / n_items are kept in the signature only for compatibility).
    genre = tf.keras.Input(shape=(), dtype=tf.int32, name="genre")
    num = tf.keras.Input(shape=(N_NUM,), dtype=tf.float32, name="num")   # user_cnt,user_avg,item_cnt,item_avg,genre_affinity,retrieval_score

    x0 = L.Concatenate()([L.Embedding(n_genres + 1, 8)(genre), num])
    mlp = lambda h: L.Dense(32, activation="relu")(L.Dropout(0.2)(L.Dense(64, activation="relu")(h)))

    if kind == "dcn":
        x = x0
        for _ in range(2):
            x = tfrs.layers.dcn.Cross()(x0, x)
        out = L.Dense(1, activation="sigmoid")(L.Concatenate()([x, mlp(x0)]))
    elif kind == "wide_deep":
        wide = L.Dense(1)(L.Concatenate()([L.Embedding(n_genres + 1, 1)(genre), num]))
        deep = L.Dense(1)(mlp(x0))
        out = L.Activation("sigmoid")(L.Add()([wide, deep]))
    else:
        raise ValueError(kind)
    return tf.keras.Model([genre, num], L.Reshape(())(out), name=f"ranker_{kind}")


def main():
    tf.random.set_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    meta = json.loads((C.PROC / "meta.json").read_text())
    feats = load_features(meta)
    feats["user_emb"] = np.load(C.ART / "user_emb.npy")
    feats["item_emb"] = np.load(C.ART / "item_emb.npy")
    (xtr, ytr), (xva, yva) = build_ranker_dataset(feats)

    for kind in KINDS:
        model = build_ranker(kind, meta["n_users"], meta["n_items"], meta["n_genres"])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                      metrics=[tf.keras.metrics.AUC(name="auc")])
        model.fit(xtr, ytr, validation_data=(xva, yva), epochs=(15 if len(ytr) < 2_000_000 else C.RANK_EPOCHS),
                  batch_size=(512 if len(ytr) < 2_000_000 else C.RANK_BATCH),
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2,
                                                              restore_best_weights=True)], verbose=2)
        model.save_weights(str(C.ART / f"ranker_{kind}.h5"))
        try:   # export for NVIDIA Triton Inference Server (see model_repository/)
            d = C.MODEL_REPO / f"ranker_{kind}"
            model.save(str(d / "1" / "model.savedmodel"), save_format="tf")
            (d / "config.pbtxt").write_text(
                f'name: "ranker_{kind}"\nplatform: "tensorflow_savedmodel"\nmax_batch_size: 4096\n')
        except Exception as e:   # non-fatal: local scoring still works
            print("Triton export skipped:", e)
    print("rankers saved")


if __name__ == "__main__":
    main()
