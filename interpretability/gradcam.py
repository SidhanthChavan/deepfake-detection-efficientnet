"""Grad-CAM visualisation for deepfake detection model."""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer='rgb_model._conv_head'):
        self.model = model
        self.gradients = None
        self.activations = None
        self.model.eval()
        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise ValueError(f"Layer {target_layer} not found.")
        layer = modules[target_layer]
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        layer.register_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))
        print(f"Grad-CAM hooked on: {target_layer}")

    def generate(self, x_rgb, x_dct, class_idx):
        self.model.zero_grad()
        out = self.model(x_rgb.unsqueeze(0), x_dct.unsqueeze(0))
        out[0, class_idx].backward()
        weights = np.mean(self.gradients[0].cpu().numpy(), axis=(1, 2))
        acts = self.activations[0].cpu().numpy()
        cam = np.maximum(np.sum(weights[:, None, None] * acts, axis=0), 0)
        cam = cv2.resize(cam, (x_rgb.shape[2], x_rgb.shape[1]))
        return cam / cam.max() if cam.max() != 0 else cam


def gradcam_visualize(model, x_rgb, x_dct, y, save_path=None):
    gc = GradCAM(model)
    class_idx = y.item() if isinstance(y, torch.Tensor) else int(y)
    cam = gc.generate(x_rgb, x_dct, class_idx)
    img = np.clip(
        x_rgb.cpu().numpy().transpose(1,2,0) *
        [0.229,0.224,0.225] + [0.485,0.456,0.406], 0, 1
    )
    heatmap = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    ) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes,
        [img, heatmap, np.clip(heatmap*0.5+img*0.5, 0, 1)],
        ['Original', 'Grad-CAM', 'Overlay']):
        ax.imshow(data)
        ax.set_title(title)
        ax.axis('off')
    label_str = 'Real' if class_idx == 0 else 'Fake'
    fig.suptitle(f'Grad-CAM — Class: {label_str}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
