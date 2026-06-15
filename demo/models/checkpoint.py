from collections import OrderedDict
from pathlib import Path

import torch

from config import DEVICE


def torch_load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}\n\n"
            "Edit the path at the top of config.py"
        )

    try:
        return torch.load(str(path), map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=DEVICE)
    except Exception:
        return torch.load(str(path), map_location=DEVICE, weights_only=False)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, OrderedDict):
        return checkpoint

    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError("Could not find a valid state_dict inside the checkpoint.")


def remove_unwanted_prefixes(state_dict):
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        for prefix in ["module.", "model.", "net."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def filter_matching_tensors(candidate, model_sd):
    filtered, skipped = OrderedDict(), []
    for k, v in candidate.items():
        if k not in model_sd:
            skipped.append((k, "not in model")); continue
        if not torch.is_tensor(v):
            skipped.append((k, "not a tensor")); continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            skipped.append((k, f"shape {tuple(v.shape)} vs {tuple(model_sd[k].shape)}")); continue
        filtered[k] = v
    return filtered, skipped
