"""Entry point for ATR clothing segmentation training and evaluation."""

from __future__ import annotations

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from config.structured import ProjectConfig
from dataset.atr_dataloader import get_dataloaders
from model.segmentation_model import build_model
import training_utils


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_dir(cfg: ProjectConfig) -> Path:
    output_root = Path(os.environ.get("OUT_DIR", cfg.run.output_root)).expanduser().resolve()
    run_dir = Path(cfg.run.run_dir) if cfg.run.run_dir else output_root / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    project_cfg: ProjectConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ProjectConfig), cfg))
    set_seed(project_cfg.run.seed)

    device_name = project_cfg.run.device
    if device_name != "cpu" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")
    print(OmegaConf.to_yaml(cfg))

    run_dir = resolve_run_dir(project_cfg)
    OmegaConf.save(cfg, run_dir / "config_saved.yaml")
    print(f"Run directory: {run_dir}")

    dataset_root = os.environ.get("ATR_DATASET_ROOT", project_cfg.dataset.root)
    class_weights_path = project_cfg.dataset.class_weights_path or str(Path(dataset_root) / "class_weights.pt")

    loaders = get_dataloaders(
        dataset_root=dataset_root,
        batch_size=project_cfg.dataloader.batch_size,
        num_workers=project_cfg.dataloader.num_workers,
        random_seed=project_cfg.run.seed,
        train_subset_fraction=project_cfg.dataset.train_subset_fraction,
    )

    model = build_model(
        model_name=project_cfg.model.name,
        encoder_name=project_cfg.model.encoder_name,
        encoder_weights=project_cfg.model.encoder_weights,
        segformer_checkpoint=project_cfg.model.segformer_checkpoint,
    ).to(device)

    if project_cfg.run.job == "train":
        training_utils.train_model(
            model=model,
            loaders=loaders,
            device=device,
            run_dir=run_dir,
            num_epochs=project_cfg.optim.num_epochs,
            lr=project_cfg.optim.lr,
            patience=project_cfg.optim.patience,
            class_weights_path=class_weights_path,
            dice_weight=project_cfg.optim.dice_weight,
            resume_path=project_cfg.checkpoint.resume,
            random_seed=project_cfg.run.seed,
            save_every_n_epochs=project_cfg.optim.save_every_n_epochs,
        )

        best_ckpt = run_dir / "checkpoints" / "best_model.pth"
        if best_ckpt.exists():
            training_utils.evaluate_model(
                model=model,
                test_loader=loaders["test"],
                device=device,
                run_dir=run_dir / "test_eval",
                checkpoint_path=best_ckpt,
            )

    elif project_cfg.run.job == "evaluate":
        checkpoint_path = project_cfg.checkpoint.resume or run_dir / "checkpoints" / "best_model.pth"
        training_utils.evaluate_model(
            model=model,
            test_loader=loaders["test"],
            device=device,
            run_dir=run_dir / "test_eval",
            checkpoint_path=Path(checkpoint_path),
        )
    else:
        raise ValueError(f"Unknown job '{project_cfg.run.job}'. Use train or evaluate.")


if __name__ == "__main__":
    main()
