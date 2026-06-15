"""
Loads all three models once at startup (called from app.py before demo.launch).
Use get_model(name) anywhere in the inference pipeline.
"""

from config import UNET_MODEL_PATH, SEGFORMER_MODEL_PATH, DEEPLAB_MODEL_PATH
from models.unet      import load_unet
from models.segformer import load_segformer
from models.deeplab   import load_deeplab

_registry: dict = {}


def load_all_models():
    print("Loading all models...")

    _registry["U-Net ResNet34"]      = load_unet(UNET_MODEL_PATH)
    _registry["SegFormer"]           = load_segformer(SEGFORMER_MODEL_PATH)
    _registry["DeepLabV3+ ResNet50"] = load_deeplab(DEEPLAB_MODEL_PATH)

    for model in _registry.values():
        model.eval()

    print("All models ready.\n")


def get_model(name: str):
    if name not in _registry:
        raise KeyError(f"Model '{name}' not found. Call load_all_models() first.")
    return _registry[name]
