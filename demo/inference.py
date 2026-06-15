import numpy as np
import torch
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMAGE_H, IMAGE_W, NUM_CLASSES, CLASS_NAMES, PALETTE, DEVICE
from models.loader import get_model


eval_transforms = A.Compose([
    A.Resize(IMAGE_H, IMAGE_W),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, NUM_CLASSES - 1)
    return PALETTE[mask]


def get_detected_classes(mask: np.ndarray) -> list[int]:
    return [int(c) for c in np.unique(mask) if 0 <= c < NUM_CLASSES]


def run_inference(pil_img: Image.Image, model_name: str):
    """
    Returns:
        overlay  : PIL Image  — original blended with color mask
        detected : list[int]  — class ids present in the prediction
    """
    model = get_model(model_name)

    original     = pil_img.convert("RGB")
    original_np  = np.array(original)
    original_size = original.size  # (W, H)

    tensor = eval_transforms(image=original_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits    = model(tensor)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    detected   = get_detected_classes(pred_mask)
    color_mask = Image.fromarray(mask_to_color(pred_mask))
    color_mask = color_mask.resize(original_size, resample=Image.NEAREST)
    overlay    = Image.blend(original, color_mask, alpha=0.45)

    return overlay, detected
