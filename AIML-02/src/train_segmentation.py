"""Step 3b: 3-D U-Net (MONAI) segmenting the hippocampus (anterior/posterior) on MRI.
Data: Medical Segmentation Decathlon, Task04_Hippocampus (260 labelled volumes)."""
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (AsDiscrete, Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd,
                              Orientationd, RandAffined, RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
                              ResizeWithPadOrCropd)
from monai.utils import set_determinism
from src import config as C

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
CLASSES = ["anterior", "posterior"]


def get_files(root):
    d = root / "Task04_Hippocampus"
    if not d.exists():
        root.mkdir(parents=True, exist_ok=True)
        download_and_extract(URL, str(root / "Task04_Hippocampus.tar"), str(root))
    meta = json.load(open(d / "dataset.json"))
    return [{"image": str(d / x["image"]), "label": str(d / x["label"])} for x in meta["training"]]


def evaluate(model, loader, device):
    post_p, post_l = AsDiscrete(argmax=True, to_onehot=3), AsDiscrete(to_onehot=3)
    metric = DiceMetric(include_background=False, reduction="mean_batch")
    model.eval()
    with torch.no_grad():
        for b in loader:
            x, y = b["image"].to(device), b["label"].to(device)
            out = model(x)
            metric(y_pred=[post_p(o) for o in decollate_batch(out)], y=[post_l(l) for l in decollate_batch(y)])
    res = metric.aggregate().cpu().numpy()
    metric.reset()
    return res


def save_examples(model, loader, device, n=3):
    out_dir = C.RESULTS / "seg_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for k, b in enumerate(loader):
            if k >= n:
                break
            pred = model(b["image"].to(device)).argmax(1)[0].cpu().numpy()
            img, gt = b["image"][0, 0].numpy(), b["label"][0, 0].numpy()
            z = int((gt > 0).sum((0, 1)).argmax())
            fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
            for a, (title, m) in zip(ax, (("MRI slice", None), ("Ground truth", gt), ("U-Net prediction", pred))):
                a.imshow(img[:, :, z].T, cmap="gray", origin="lower")
                if m is not None:
                    a.imshow(np.ma.masked_where(m[:, :, z].T == 0, m[:, :, z].T), cmap="cool", alpha=0.6,
                             origin="lower", vmin=0, vmax=2)
                a.set_title(title)
                a.axis("off")
            fig.suptitle("Hippocampus segmentation - held-out case " + str(k + 1) + "  (" + C.DISCLAIMER + ")", fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / f"case_{k+1}.png", dpi=120)
            plt.close(fig)


def main():
    cfg = C.load_params()["segmentation"]
    set_determinism(seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = get_files(C.DATA / "msd")
    random.Random(0).shuffle(files)
    n = len(files)
    train_f, val_f, test_f = files[:int(.7 * n)], files[int(.7 * n):int(.85 * n)], files[int(.85 * n):]
    print("volumes train/val/test:", len(train_f), len(val_f), len(test_f))

    keys, sz = ["image", "label"], tuple(cfg["spatial_size"])
    base = [LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
            NormalizeIntensityd("image", nonzero=True, channel_wise=True), ResizeWithPadOrCropd(keys, spatial_size=sz)]
    train_t = Compose(base + [
        RandAffined(keys, prob=0.5, rotate_range=(0.2, 0.2, 0.2), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandFlipd(keys, prob=0.5, spatial_axis=0), RandScaleIntensityd("image", factors=0.1, prob=0.5),
        RandShiftIntensityd("image", offsets=0.1, prob=0.5), EnsureTyped(keys)])
    eval_t = Compose(base + [EnsureTyped(keys)])
    train_l = DataLoader(CacheDataset(train_f, train_t, cache_rate=1.0, num_workers=2), batch_size=cfg["batch_size"], shuffle=True)
    val_l = DataLoader(CacheDataset(val_f, eval_t, cache_rate=1.0, num_workers=2), batch_size=2)
    test_l = DataLoader(CacheDataset(test_f, eval_t, cache_rate=1.0, num_workers=2), batch_size=1)

    model = UNet(spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64, 128), strides=(2, 2, 2),
                 num_res_units=2).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), cfg["lr"])
    best, best_state, history = -1.0, None, []
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for b in train_l:
            x, y = b["image"].to(device), b["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss)
        total /= len(train_l)
        if (epoch + 1) % 2 == 0 or epoch + 1 == cfg["epochs"]:
            d = evaluate(model, val_l, device)
            history.append({"epoch": epoch + 1, "train_loss": total, "val_dice_mean": float(d.mean())})
            print(f"epoch {epoch+1}/{cfg['epochs']} loss {total:.4f}  val Dice {d.round(4)} mean {d.mean():.4f}")
            if d.mean() > best:
                best = float(d.mean())
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_d = evaluate(model, test_l, device)
    C.MODELS.mkdir(exist_ok=True)
    torch.save(best_state, C.MODELS / "unet3d_hippocampus.pt")
    metrics = {"dataset": "MSD Task04_Hippocampus", "n_train": len(train_f), "n_val": len(val_f), "n_test": len(test_f),
               "best_val_dice_mean": best, "test_dice_mean": float(test_d.mean()),
               "test_dice_per_class": dict(zip(CLASSES, [float(v) for v in test_d])), "history": history}
    (C.RESULTS / "seg_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("TEST Dice:", metrics["test_dice_per_class"], "mean", round(metrics["test_dice_mean"], 4))
    save_examples(model, test_l, device)


if __name__ == "__main__":
    main()
