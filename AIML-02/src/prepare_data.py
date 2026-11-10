"""Step 1-2: get NIH ChestX-ray14 (via MedMNIST 'ChestMNIST'), subsample, store as compressed arrays.
Preprocessing at train time: ImageNet normalisation + GPU augmentation (see train_classifier.py)."""
import json
import numpy as np
from medmnist import ChestMNIST
from src import config as C


def load(split, size, max_n, rng):
    root = C.DATA / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    ds = ChestMNIST(split=split, download=True, root=str(root), size=size)
    imgs, labels = np.asarray(ds.imgs), np.asarray(ds.labels)
    if imgs.ndim == 4:
        imgs = imgs[..., 0]
    if max_n and len(imgs) > max_n:
        idx = np.sort(rng.choice(len(imgs), max_n, replace=False))
        imgs, labels = imgs[idx], labels[idx]
    return imgs.astype(np.uint8), labels.astype(np.uint8)


def main():
    p = C.load_params()["data"]
    rng = np.random.default_rng(p["seed"])
    C.PROC.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {"source": "NIH ChestX-ray14 via MedMNIST v3 (ChestMNIST, down-sampled)", "image_size": p["image_size"]}
    for split, key in (("train", "max_train"), ("val", "max_val"), ("test", "max_test")):
        imgs, labels = load(split, p["image_size"], p[key], rng)
        np.savez_compressed(C.PROC / f"chest_{split}.npz", imgs=imgs, labels=labels)
        summary[split] = {"n": int(len(imgs)),
                          "no_finding_fraction": float((labels.sum(1) == 0).mean()),
                          "prevalence": {n: round(float(labels[:, i].mean()), 4) for i, n in enumerate(C.LABELS)}}
        print(split, imgs.shape, labels.shape)
    (C.RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
