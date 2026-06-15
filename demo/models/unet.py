import segmentation_models_pytorch as smp

from config import (
    UNET_ENCODER_NAME, UNET_ENCODER_WEIGHTS,
    NUM_CLASSES, DEVICE,
)
from models.checkpoint import torch_load_checkpoint, extract_state_dict, remove_unwanted_prefixes


def build_unet():
    model = smp.Unet(
        encoder_name=UNET_ENCODER_NAME,
        encoder_weights=UNET_ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model


def load_unet(checkpoint_path):
    model = build_unet().to(DEVICE)

    ckpt       = torch_load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(ckpt)
    state_dict = remove_unwanted_prefixes(state_dict)

    model.load_state_dict(state_dict, strict=True)
    print(f"[UNet] loaded from {checkpoint_path}")
    return model
