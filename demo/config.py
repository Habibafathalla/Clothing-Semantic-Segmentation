import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_H = 512
IMAGE_W = 384

# ---- checkpoint paths ----
UNET_MODEL_PATH      = r"Unet Models\Backbone_resnet34_Dataset saved with (preprocessing+Aug)\best_model.pth"
SEGFORMER_MODEL_PATH = r"SegTransformer Models\Saved Dataset with preprcoessing and Augmentation\best_model.pth"
DEEPLAB_MODEL_PATH   = r"DeepLabV3 Models\Backbone _resnet50_Dataset saved withpreprocessing and aug while training\best_model.pth"

# ---- shared 8-class schema (UNet now also uses this) ----
NUM_CLASSES  = 8
CLASS_NAMES  = [
    "Background", "Upper-clothes", "Skirt",
    "Pants", "Dress", "Shoes", "Human Parts", "Accessories",
]

ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}

# ---- encoder settings ----
UNET_ENCODER_NAME       = "resnet34"
UNET_ENCODER_WEIGHTS    = "imagenet"
DEEPLAB_ENCODER_NAME    = "resnet50"
DEEPLAB_ENCODER_WEIGHTS = "imagenet"
SEGFORMER_BASE_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

# ---- color palette (shared by all 3 models — same 8 classes) ----
PALETTE = np.array([
    [0,   0,   0  ],  # Background
    [170, 0,   51 ],  # Upper-clothes
    [255, 85,  0  ],  # Skirt
    [0,   0,   85 ],  # Pants
    [0,   119, 221],  # Dress
    [85,  51,  0  ],  # Shoes
    [52,  86,  128],  # Human Parts
    [170, 255, 85 ],  # Accessories
], dtype=np.uint8)

# ---- model registry (used by UI dropdown and loader) ----
MODEL_NAMES = ["U-Net ResNet34", "SegFormer", "DeepLabV3+ ResNet50"]
