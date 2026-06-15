"""Losses, metrics, and checkpoint helpers."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_loss(class_weights: torch.Tensor, dice_weight: float = 0.5):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ce_loss(predictions, targets) + dice_weight * dice_loss(predictions, targets)

    return compute_loss


def compute_miou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):
    pred_classes = predictions.argmax(dim=1)
    iou_per_class = []

    for class_id in range(num_classes):
        predicted_as_class = pred_classes == class_id
        actually_class = targets == class_id
        intersection = (predicted_as_class & actually_class).sum().float()
        union = (predicted_as_class | actually_class).sum().float()
        if union > 0:
            iou_per_class.append((intersection / union).item())
        else:
            iou_per_class.append(float("nan"))

    present = [v for v in iou_per_class if not np.isnan(v)]
    mean_iou = sum(present) / len(present) if present else 0.0
    return mean_iou, iou_per_class


def update_confusion_matrix(conf_matrix: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):
    pred_classes = predictions.argmax(dim=1)
    pred_flat = pred_classes.view(-1).cpu()
    target_flat = targets.view(-1).cpu()
    valid_mask = target_flat < num_classes
    pred_flat = pred_flat[valid_mask]
    target_flat = target_flat[valid_mask]
    cell_indices = num_classes * target_flat + pred_flat
    conf_matrix += torch.bincount(cell_indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf_matrix


def get_metrics_from_confusion_matrix(conf_matrix: torch.Tensor):
    cm = conf_matrix.float()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "pixel_accuracy": (tp.sum() / cm.sum()).item(),
        "mean_accuracy": (tp / (cm.sum(dim=1) + 1e-6)).mean().item(),
        "miou": iou.mean().item(),
        "iou_per_class": iou.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def find_latest_checkpoint(path: str | Path | None) -> str | None:
    if path is None:
        return None

    path = str(path)
    if os.path.isfile(path) and path.endswith(".pth"):
        return path

    if os.path.isdir(path):
        epoch_files = glob.glob(os.path.join(path, "epoch_*.pth"))
        if epoch_files:
            def get_epoch_num(file_path):
                name = os.path.basename(file_path)
                match = re.search(r"epoch_(\d+)\.pth$", name)
                return int(match.group(1)) if match else -1

            return sorted(epoch_files, key=get_epoch_num)[-1]

        best = os.path.join(path, "best_model.pth")
        if os.path.exists(best):
            return best

    return None


def load_class_weights(weights_path: str | Path, device: torch.device, num_classes: int) -> torch.Tensor | None:
    weights_path = Path(weights_path)
    if weights_path.exists():
        print(f"Loading class weights from {weights_path}")
        return torch.load(weights_path, map_location=device)
    return None


def compute_class_weights_from_loader(train_loader, num_classes: int, device: torch.device):
    from tqdm import tqdm

    class_pixel_counts = torch.zeros(num_classes)
    for batch in tqdm(train_loader, desc="Counting pixels per class"):
        masks = batch["mask"]
        for class_id in range(num_classes):
            class_pixel_counts[class_id] += (masks == class_id).sum()

    class_freq = class_pixel_counts / class_pixel_counts.sum()
    class_weights = 1.0 / torch.sqrt(class_freq + 1e-6)
    class_weights = class_weights / class_weights.sum()
    return class_weights.to(device)
