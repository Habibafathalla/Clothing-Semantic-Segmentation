"""Training, validation, and evaluation loops."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from dataset.labels import CLASS_NAMES, NUM_CLASSES
from utils.metrics import (
    build_loss,
    compute_class_weights_from_loader,
    compute_miou,
    find_latest_checkpoint,
    get_metrics_from_confusion_matrix,
    load_class_weights,
    update_confusion_matrix,
)


def poly_lr_schedule_factory(total_steps: int):
    def poly_lr_schedule(current_step: int) -> float:
        progress = current_step / max(total_steps, 1)
        return max(0.0, 1.0 - progress) ** 0.9

    return poly_lr_schedule


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Saved checkpoint -> {path}")


def plot_training_curves(train_losses, val_losses, train_mious, val_mious, output_path: Path, title: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train", marker="o", markersize=3)
    ax1.plot(val_losses, label="Val", marker="o", markersize=3)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(train_mious, label="Train", marker="o", markersize=3)
    ax2.plot(val_mious, label="Val", marker="o", markersize=3)
    ax2.set_title("mIoU")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curves -> {output_path}")


def plot_confusion_matrix(conf_matrix: torch.Tensor, class_names, output_path: Path, normalize: bool = True):
    import itertools

    cm = conf_matrix.float().numpy()
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)
        fmt = ".2f"
    else:
        fmt = ".0f"

    fig, ax = plt.subplots(figsize=(11, 9))
    img = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    plt.colorbar(img, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (row-normalised)" if normalize else "Confusion Matrix (counts)")

    threshold = cm.max() / 2.0
    for row, col in itertools.product(range(len(class_names)), range(len(class_names))):
        ax.text(
            col,
            row,
            format(cm[row, col], fmt),
            ha="center",
            va="center",
            fontsize=6,
            color="white" if cm[row, col] > threshold else "black",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix -> {output_path}")


def prepare_training(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    lr: float,
    num_epochs: int,
    class_weights_path: str | Path | None,
    dice_weight: float,
):
    class_weights = load_class_weights(class_weights_path, device, NUM_CLASSES)
    if class_weights is None:
        print("Class weights not found — computing from training loader...")
        class_weights = compute_class_weights_from_loader(train_loader, NUM_CLASSES, device)
        if class_weights_path is not None:
            Path(class_weights_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(class_weights.cpu(), class_weights_path)
            print(f"Saved class weights -> {class_weights_path}")

    print("Class weights:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15}: {class_weights[i].item():.4f}")

    compute_loss = build_loss(class_weights, dice_weight=dice_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = num_epochs * len(train_loader)
    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_schedule_factory(total_steps))
    scaler = GradScaler(device.type)
    print(f"Total training steps: {total_steps:,}")
    return compute_loss, optimizer, scheduler, scaler


def maybe_resume_training(
    model,
    optimizer,
    scheduler,
    resume_path: str | Path | None,
    device: torch.device,
):
    state = {
        "start_epoch": 1,
        "best_miou": 0.0,
        "train_losses": [],
        "val_losses": [],
        "train_mious": [],
        "val_mious": [],
        "epochs_no_improve": 0,
    }

    checkpoint_path = find_latest_checkpoint(resume_path)
    if checkpoint_path is None:
        print("No checkpoint found — starting from scratch.")
        return state

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    state["start_epoch"] = ckpt["epoch"] + 1
    state["best_miou"] = ckpt.get("best_miou", 0.0)
    state["train_losses"] = ckpt.get("train_losses", [])
    state["val_losses"] = ckpt.get("val_losses", [])
    state["train_mious"] = ckpt.get("train_mious", [])
    state["val_mious"] = ckpt.get("val_mious", [])
    state["epochs_no_improve"] = ckpt.get("epochs_no_improve", 0)

    print(f"Resumed from  : {checkpoint_path}")
    print(f"Continuing at : epoch {state['start_epoch']} | best mIoU so far: {state['best_miou']:.4f}")
    return state


def train_model(
    model,
    loaders,
    device: torch.device,
    run_dir: Path,
    num_epochs: int,
    lr: float,
    patience: int,
    class_weights_path: str | Path | None,
    dice_weight: float,
    resume_path: str | Path | None,
    random_seed: int,
    save_every_n_epochs: int = 3,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    compute_loss, optimizer, scheduler, scaler = prepare_training(
        model=model,
        train_loader=loaders["train"],
        device=device,
        lr=lr,
        num_epochs=num_epochs,
        class_weights_path=class_weights_path,
        dice_weight=dice_weight,
    )

    state = maybe_resume_training(model, optimizer, scheduler, resume_path, device)
    best_ckpt_path = checkpoint_dir / "best_model.pth"

    for epoch in range(state["start_epoch"], num_epochs + 1):
        torch.manual_seed(random_seed + epoch)
        model.train()
        epoch_loss = 0.0
        epoch_miou = 0.0

        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}/{num_epochs} [train]"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                predictions = model(images)
                loss = compute_loss(predictions, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            miou, _ = compute_miou(predictions.detach(), masks, NUM_CLASSES)
            epoch_loss += loss.item()
            epoch_miou += miou

        state["train_losses"].append(epoch_loss / len(loaders["train"]))
        state["train_mious"].append(epoch_miou / len(loaders["train"]))

        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        per_class_iou_sum = [0.0] * NUM_CLASSES
        per_class_iou_count = [0] * NUM_CLASSES

        with torch.no_grad():
            for batch in tqdm(loaders["validation"], desc=f"Epoch {epoch}/{num_epochs} [val]  "):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                with autocast(device_type=device.type):
                    predictions = model(images)
                    loss = compute_loss(predictions, masks)

                miou, class_ious = compute_miou(predictions, masks, NUM_CLASSES)
                val_loss += loss.item()
                val_miou += miou
                for class_id, iou in enumerate(class_ious):
                    if not np.isnan(iou):
                        per_class_iou_sum[class_id] += iou
                        per_class_iou_count[class_id] += 1

        state["val_losses"].append(val_loss / len(loaders["validation"]))
        state["val_mious"].append(val_miou / len(loaders["validation"]))

        print(
            f"\nEpoch {epoch:02d} | "
            f"Train Loss: {state['train_losses'][-1]:.4f}  Val Loss: {state['val_losses'][-1]:.4f} | "
            f"Train mIoU: {state['train_mious'][-1]:.4f}  Val mIoU: {state['val_mious'][-1]:.4f}"
        )

        if epoch % 5 == 0:
            print("  Per-class IoU:")
            for c in range(NUM_CLASSES):
                avg = (
                    per_class_iou_sum[c] / per_class_iou_count[c]
                    if per_class_iou_count[c] > 0
                    else 0.0
                )
                print(f"    {CLASS_NAMES[c]:<15}: {avg:.4f}")

        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": state["best_miou"],
            "train_losses": state["train_losses"],
            "val_losses": state["val_losses"],
            "train_mious": state["train_mious"],
            "val_mious": state["val_mious"],
            "epochs_no_improve": state["epochs_no_improve"],
        }

        if state["val_mious"][-1] > state["best_miou"]:
            state["best_miou"] = state["val_mious"][-1]
            ckpt_dict["best_miou"] = state["best_miou"]
            state["epochs_no_improve"] = 0
            save_checkpoint(best_ckpt_path, ckpt_dict)
            print(f"Best model saved (val mIoU = {state['best_miou']:.4f})")
        else:
            state["epochs_no_improve"] += 1
            print(f"  No improvement: {state['epochs_no_improve']}/{patience} epochs")
            if state["epochs_no_improve"] >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

        if epoch % save_every_n_epochs == 0:
            periodic_path = checkpoint_dir / f"epoch_{epoch:02d}.pth"
            ckpt_dict["epochs_no_improve"] = state["epochs_no_improve"]
            save_checkpoint(periodic_path, ckpt_dict)

    plot_training_curves(
        state["train_losses"],
        state["val_losses"],
        state["train_mious"],
        state["val_mious"],
        run_dir / "training_curves.png",
        title="Training Curves",
    )

    history = {
        "train_losses": state["train_losses"],
        "val_losses": state["val_losses"],
        "train_mious": state["train_mious"],
        "val_mious": state["val_mious"],
        "best_miou": state["best_miou"],
    }
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return state


def evaluate_model(model, test_loader, device: torch.device, run_dir: Path, checkpoint_path: Path | None = None):
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint for evaluation: {checkpoint_path}")

    model.eval()
    conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with autocast(device_type=device.type):
                predictions = model(images)
            conf_matrix = update_confusion_matrix(conf_matrix, predictions, masks, NUM_CLASSES)

    metrics = get_metrics_from_confusion_matrix(conf_matrix)

    print(f"\n{'=' * 55}")
    print(f"  Pixel Accuracy      : {metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"  Mean Class Accuracy : {metrics['mean_accuracy'] * 100:.2f}%")
    print(f"  Mean IoU (mIoU)     : {metrics['miou'] * 100:.2f}%")
    print(f"{'=' * 55}")
    print(f"\n{'Class':<16} {'IoU':>7} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 55)
    for c in range(NUM_CLASSES):
        print(
            f"  {CLASS_NAMES[c]:<14} "
            f"{metrics['iou_per_class'][c] * 100:>6.2f}%  "
            f"{metrics['precision'][c] * 100:>8.2f}%  "
            f"{metrics['recall'][c] * 100:>6.2f}%  "
            f"{metrics['f1'][c] * 100:>6.2f}%"
        )

    plot_confusion_matrix(conf_matrix, CLASS_NAMES, run_dir / "confusion_matrix.png", normalize=True)

    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
