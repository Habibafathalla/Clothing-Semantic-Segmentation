from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunConfig:
    job: str = "train"  # train | evaluate
    name: str = "atr_seg_unet"
    output_root: str = "./outputs"
    run_dir: Optional[str] = None
    seed: int = 42
    device: str = "cuda"


@dataclass
class DatasetConfig:
    root: str = "./data/atr_merged_preprocessed"
    hf_dataset_id: str = "ckotait/ATRDataset"
    class_weights_path: Optional[str] = None
    train_subset_fraction: float = 1.0


@dataclass
class DataloaderConfig:
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class ModelConfig:
    name: str = "unet"  # unet | deeplabv3plus | segformer
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512"


@dataclass
class OptimConfig:
    lr: float = 6e-5
    num_epochs: int = 50
    dice_weight: float = 0.5
    patience: int = 7
    save_every_n_epochs: int = 3


@dataclass
class CheckpointConfig:
    resume: Optional[str] = None


@dataclass
class ProjectConfig:
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
