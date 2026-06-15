"""Segmentation model factory for U-Net, DeepLabV3+, and SegFormer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from dataset.labels import CLASS_NAMES, MERGED_LABEL2ID, NUM_CLASSES


class SegFormerWrapper(nn.Module):
    """Wrap HuggingFace SegFormer and upsample logits to input resolution."""

    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        config = SegformerConfig.from_pretrained(
            model_name,
            num_labels=NUM_CLASSES,
            id2label={i: name for i, name in enumerate(CLASS_NAMES)},
            label2id=MERGED_LABEL2ID,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=images).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def build_model(
    model_name: str,
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    segformer_checkpoint: str = "nvidia/segformer-b0-finetuned-ade-512-512",
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    elif model_name in {"deeplabv3plus", "deeplab", "deeplabv3+"}:
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    elif model_name == "segformer":
        model = SegFormerWrapper(model_name=segformer_checkpoint)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: unet, deeplabv3plus, segformer."
        )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_name} | encoder={encoder_name if model_name != 'segformer' else segformer_checkpoint}")
    print(f"Parameters: {total_params:.1f}M")
    return model
