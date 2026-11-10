"""Step 6 - the online two-stage recommender: FAISS retrieval -> deep ranker -> Redis cache,
plus cold-start fallbacks (popularity / genre-based content filter)."""
import json, time, hashlib
import numpy as np
import pandas as pd
from src import config as C
from src.features import load_features
from src.cache import Cache


class Recommender:
    def __init__(self, seen_splits=("train", "val", "test"), load_models=True):
        self.meta = json.loads((C.PROC / "meta.json").read_text())
        self.feats = load_features(self.meta)
        self.movies = self.feats["movies"]                         # indexed by item_idx
        self.user_map = {int(k): v for k, v in self.meta["user_map"].items()}
        self.n_items = self.meta["n_items"]
        self.cache = Cache()
        self.seen = self._load_seen(seen_splits)
        self.pop_order = np.argsort(-self.feats["item_pos"][1:], kind="stable") + 1   # item_idx, best first
        self._genre_sets = {i: set(g.split("|")) for i, g in self.movies.genres.items()}
        self.user_emb = self.index = None
        self._rankers = {}
        self._triton = {}
        if load_models:
            import faiss
            self.user_emb = np.load(C.ART / "user_emb.npy")
            self.index = faiss.read_index(str(C.ART / "items.faiss"))
            self.feats["user_emb"], self.feats["item_emb"] = self.user_emb, np.load(C.ART / "item_emb.npy")

    # ---------- helpers
    def _load_seen(self, splits):
        frames = [pd.read_csv(C.PROC / f"{s}.csv", usecols=["user_idx", "item_idx"]) for s in splits]
        df = pd.concat(frames)
        df = df[(df.user_idx > 0) & (df.item_idx > 0)]
        return {u: set(g.item_idx.values) for u, g in df.groupby("user_idx")}

    def popular(self, exclude=(), k=10):
        out = []
        for it in self.pop_order:
            if it not in exclude:
                out.append(int(it))
                if len(out) == k:
                    break
        return out

    def retrieve(self, uidx, n=C.N_CANDIDATES):
        """Stage 1: nearest items to the user embedding (FAISS), minus already-seen items."""
        seen = self.seen.get(uidx, set())
        k = min(self.n_items, n + len(seen))
        scores, ids = self.index.search(self.user_emb[uidx:uidx + 1], k)
        items = ids[0] + 1
        keep = [(int(i), float(s)) for i, s in zip(items, scores[0]) if i not in seen][:n]
        return np.array([i for i, _ in keep]), np.array([s for _, s in keep])

    def _ranker(self, kind):
        if kind not in self._rankers:
            from src.train_ranker import build_ranker
            m = build_ranker(kind, self.meta["n_users"], self.meta["n_items"], self.meta["n_genres"])
            m.load_weights(str(C.ART / f"ranker_{kind}.h5"))
            self._rankers[kind] = m
        return self._rankers[kind]

    def score_pairs(self, kind, users, items):
        """Stage 2: ranker probability for (user,item) pairs. Uses Triton if TRITON_URL is set."""
        from src.ranker_data import make_inputs
        x = make_inputs(np.asarray(users), np.asarray(items), self.feats)
        if C.TRITON_URL:
            try:
                return self._score_triton(kind, x)
            except Exception as e:
                print("Triton unavailable, scoring locally:", e)
        return self._ranker(kind).predict(x, batch_size=8192, verbose=0).ravel()

    def _score_triton(self, kind, x):
        import tritonclient.http as hc
        name = f"ranker_{kind}"
        if name not in self._triton:
            c = hc.InferenceServerClient(C.TRITON_URL)
            self._triton[name] = (c, c.get_model_metadata(name)["outputs"][0]["name"])
        c, out = self._triton[name]
        inputs = []
        for k, arr in x.items():
            t = hc.InferInput(k, list(arr.shape), "INT32" if arr.dtype == np.int32 else "FP32")
            t.set_data_from_numpy(arr)
            inputs.append(t)
        return c.infer(name, inputs, outputs=[hc.InferRequestedOutput(out)]).as_numpy(out).ravel()

    # ---------- public API
    def _fmt(self, items, scores):
        rows = []
        for it, s in zip(items, scores):
            r = self.movies.loc[int(it)]
            rows.append({"movieId": int(r.movieId), "title": r.title, "genres": r.genres, "score": round(float(s), 4)})
        return rows

    def recommend(self, user_id=None, k=10, ranker="dcn", genres=None, use_cache=True):
        t0 = time.time()
        uidx = self.user_map.get(int(user_id)) if user_id not in (None, "") else None
        n_hist = len(self.seen.get(uidx, ())) if uidx else 0
        key = "rec:" + hashlib.md5(f"{uidx}|{k}|{ranker}|{sorted(genres or [])}".encode()).hexdigest()
        if use_cache:
            hit = self.cache.get(key)
            if hit:
                hit["cache_hit"], hit["latency_ms"] = True, round((time.time() - t0) * 1000, 2)
                return hit

        if uidx is None or n_hist < C.MIN_HISTORY:          # ---- cold start
            if genres:
                want = set(genres)
                cand = [int(i) for i in self.pop_order if self._genre_sets[int(i)] & want][:k * 5]
                cand.sort(key=lambda i: -len(self._genre_sets[i] & want))      # best genre overlap first
                items = cand[:k]
                strategy = "cold-start: content-based (genres) + popularity"
            else:
                items, strategy = self.popular(self.seen.get(uidx, set()), k), "cold-start: popularity"
            scores = np.linspace(1, 0.5, len(items))
        else:                                                # ---- two-stage
            cands, _ = self.retrieve(uidx)
            probs = self.score_pairs(ranker, np.full(len(cands), uidx), cands)
            order = np.argsort(-probs)[:k]
            items, scores = cands[order], probs[order]
            strategy = f"two-stage: FAISS retrieval -> {ranker} ranker"
        out = {"strategy": strategy, "items": self._fmt(items, scores), "cache_hit": False}
        out["latency_ms"] = round((time.time() - t0) * 1000, 2)
        if use_cache:
            self.cache.set(key, out)
        return out
