import segmentation_models_pytorch as smp

from config import (
    DEEPLAB_ENCODER_NAME, DEEPLAB_ENCODER_WEIGHTS,
    NUM_CLASSES, DEVICE,
)
from models.checkpoint import torch_load_checkpoint, extract_state_dict, remove_unwanted_prefixes


def build_deeplab():
    model = smp.DeepLabV3Plus(
        encoder_name=DEEPLAB_ENCODER_NAME,
        encoder_weights=DEEPLAB_ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model


def load_deeplab(checkpoint_path):
    model = build_deeplab().to(DEVICE)

    ckpt       = torch_load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(ckpt)
    state_dict = remove_unwanted_prefixes(state_dict)

    model.load_state_dict(state_dict, strict=True)
    print(f"[DeepLab] loaded from {checkpoint_path}")
    return model
