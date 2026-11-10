"""Writes demo DICOM studies (from held-out test images) + a Grad-CAM example figure."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C
from src.dicom_utils import gray_to_dicom, save_dicom
from src.inference import ChestClassifier


def ref_text(lbl):
    names = [C.LABELS[j] for j in np.where(lbl)[0]]
    return ", ".join(names) if names else "no finding"


def main():
    n = C.load_params()["examples"]["n_dicom"]
    te = np.load(C.PROC / "chest_test.npz")
    imgs, labels = te["imgs"], te["labels"]
    rng = np.random.default_rng(0)
    pos = rng.choice(np.where(labels.sum(1) > 0)[0], n * 2 // 3, replace=False)
    neg = rng.choice(np.where(labels.sum(1) == 0)[0], n - len(pos), replace=False)
    pick = [int(i) for i in np.concatenate([pos, neg])]

    out = C.SAMPLES / "dicom"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, i in enumerate(pick):
        name = f"demo_{k:02d}.dcm"
        ds = gray_to_dicom(imgs[i], patient_id=f"DEMO{k:04d}", comments=f"Dataset reference label: {ref_text(labels[i])}")
        save_dicom(ds, out / name)
        rows.append([name, ref_text(labels[i])])
    with open(C.SAMPLES / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([["file", "reference_label"]] + rows)

    clf = ChestClassifier()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.5))
    for slot, i in enumerate(pick[:8]):
        r = clf.predict(imgs[i])
        ax1, ax2 = axes[slot // 2, (slot % 2) * 2], axes[slot // 2, (slot % 2) * 2 + 1]
        ax1.imshow(imgs[i], cmap="gray")
        ax1.set_title("input\nref: " + ref_text(labels[i]), fontsize=8)
        ax2.imshow(r["overlay"])
        ax2.set_title(f"Grad-CAM: {r['top_label']} ({r['confidence']:.2f})", fontsize=8)
        ax1.axis("off")
        ax2.axis("off")
    fig.suptitle("Grad-CAM on held-out test images (" + C.DISCLAIMER + ")", fontsize=11)
    plt.tight_layout()
    plt.savefig(C.RESULTS / "gradcam_examples.png", dpi=130)
    print("wrote", len(pick), "DICOM files to", out)


if __name__ == "__main__":
    main()
