"""Step 3a: transfer-learning classifier (DenseNet-121, ImageNet weights) for 14 chest findings."""
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C
from src.model import build_model
from src.preprocess import normalize_batch


def augment(x):
    """Light, anatomy-safe augmentation on a (B,1,H,W) batch in [0,1]: small rotation/scale/shift + brightness/contrast.
    (No horizontal flips: left/right matters in chest X-rays, e.g. heart position.)"""
    B, dev = x.shape[0], x.device
    r = lambda: torch.rand(B, device=dev) * 2 - 1
    ang, sc = r() * (10 * np.pi / 180), 1 + r() * 0.1
    tx, ty = r() * 0.05, r() * 0.05
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False),
                      padding_mode="border", align_corners=False)
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.08
    return ((x - 0.5) * c + 0.5 + b).clamp(0, 1)


@torch.no_grad()
def predict(model, imgs, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(device).float().div(255).unsqueeze(1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(torch.sigmoid(model(normalize_batch(x))).float().cpu().numpy())
    return np.concatenate(out)


def per_class_auc(y, p):
    return [float(roc_auc_score(y[:, i], p[:, i])) if 0 < y[:, i].sum() < len(y) else None for i in range(y.shape[1])]


def mean_of(vals):
    v = [a for a in vals if a is not None]
    return float(np.mean(v)) if v else 0.0


def main():
    cfg = C.load_params()["classifier"]
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    tr, va, te = (np.load(C.PROC / f"chest_{s}.npz") for s in ("train", "val", "test"))
    xtr, ytr, xva, yva, xte, yte = tr["imgs"], tr["labels"], va["imgs"], va["labels"], te["imgs"], te["labels"]

    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = cfg["epochs"] * int(np.ceil(len(xtr) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()

    best, best_state, history = -1.0, None, []
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = np.random.permutation(len(xtr))
        t0, total = time.time(), 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = np.sort(perm[i:i + cfg["batch_size"]])
            x = torch.from_numpy(xtr[idx]).to(device).float().div(255).unsqueeze(1)
            y = torch.from_numpy(ytr[idx]).to(device).float()
            x = normalize_batch(augment(x))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss) * len(idx)
        val_auc = mean_of(per_class_auc(yva, predict(model, xva, device)))
        history.append({"epoch": epoch + 1, "train_loss": total / len(xtr), "val_mean_auroc": val_auc})
        print(f"epoch {epoch+1}/{cfg['epochs']}  loss {total/len(xtr):.4f}  val mean AUROC {val_auc:.4f}  ({time.time()-t0:.0f}s)")
        if val_auc > best:
            best = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_auc = per_class_auc(yte, predict(model, xte, device))
    C.MODELS.mkdir(exist_ok=True)
    C.RESULTS.mkdir(exist_ok=True)
    torch.save(best_state, C.MODELS / "densenet121_chestmnist.pt")
    (C.MODELS / "classifier_meta.json").write_text(json.dumps(
        {"version": C.MODEL_VERSION, "labels": C.LABELS, "image_size": C.load_params()["data"]["image_size"],
         "backbone": "densenet121 (ImageNet pre-trained)", "cam_layer": C.CAM_LAYER}, indent=2))
    metrics = {"version": C.MODEL_VERSION, "n_train": int(len(xtr)), "n_val": int(len(xva)), "n_test": int(len(xte)),
               "epochs": cfg["epochs"], "best_val_mean_auroc": best, "test_mean_auroc": mean_of(test_auc),
               "test_auroc_per_class": dict(zip(C.LABELS, test_auc)), "history": history}
    (C.RESULTS / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("TEST mean AUROC:", round(metrics["test_mean_auroc"], 4))

    vals = [a if a is not None else 0 for a in test_auc]
    plt.figure(figsize=(9, 4.5))
    plt.bar(C.LABELS, vals, color="#3b6fb6")
    plt.axhline(0.5, color="gray", ls="--", lw=1)
    plt.xticks(rotation=60, ha="right")
    plt.ylim(0.3, 1.0)
    plt.ylabel("AUROC (test)")
    plt.title(f"Per-finding AUROC - mean {metrics['test_mean_auroc']:.3f}")
    plt.tight_layout()
    plt.savefig(C.RESULTS / "classifier_auroc.png", dpi=150)


if __name__ == "__main__":
    main()
