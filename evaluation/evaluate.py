"""Evaluation module — metrics, ROC curve, confusion matrix."""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


def test_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x_rgb, x_dct, y in loader:
            x_rgb, x_dct = x_rgb.to(device), x_dct.to(device)
            out = model(x_rgb, x_dct)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_preds.append(out.argmax(1).cpu())
            all_labels.append(y)
            all_probs.extend(probs)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    print(classification_report(labels, preds, target_names=['Real', 'Fake']))
    print('Confusion Matrix:\n', confusion_matrix(labels, preds))
    if len(set(labels)) == 2:
        print(f'ROC AUC: {roc_auc_score(labels, all_probs):.4f}')
    return preds, labels, all_probs


def plot_roc_curve(labels, probs, save_path=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
