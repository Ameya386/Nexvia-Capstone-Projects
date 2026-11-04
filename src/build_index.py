"""Step 4 - index item embeddings in FAISS for fast nearest-neighbour search.
Embeddings are L2-normalised, so inner product == cosine similarity.

    python -m src.build_index
"""
import numpy as np
import faiss
from src import config as C


def main():
    iemb = np.load(C.ART / "item_emb.npy")
    index = faiss.IndexFlatIP(iemb.shape[1])   # exact search; swap for IndexHNSWFlat on much larger catalogues
    index.add(iemb[1:])                        # row 0 = "unknown item", not indexed -> faiss id + 1 = item_idx
    faiss.write_index(index, str(C.ART / "items.faiss"))
    print("FAISS index with", index.ntotal, "items")


if __name__ == "__main__":
    main()
