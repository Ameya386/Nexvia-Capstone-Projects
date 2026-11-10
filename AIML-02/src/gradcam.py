"""Step 4: Grad-CAM (Selvaraju et al., 2017) implemented from scratch with PyTorch hooks.
Shows which image regions pushed the model's score for a finding up."""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib


class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.acts = self.grads = None
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        """x: normalised (1,3,H,W). Returns (cam in [0,1] as HxW numpy, sigmoid probabilities 1-D numpy)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            c = int(logits[0].argmax()) if class_idx is None else int(class_idx)
            logits[0, c].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
        return cam, torch.sigmoid(logits[0]).detach().cpu().numpy(), c


def overlay(gray_uint8, cam, alpha=0.55):
    """Blend a jet heatmap onto the grayscale image; low activations stay transparent."""
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    base = np.repeat(gray_uint8[..., None], 3, axis=-1).astype(np.float32)
    w = (alpha * cam)[..., None]
    return ((1 - w) * base + w * heat).clip(0, 255).astype(np.uint8)
