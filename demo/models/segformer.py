from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from config import (
    SEGFORMER_BASE_MODEL_ID, NUM_CLASSES,
    ID2LABEL, LABEL2ID, DEVICE,
)
from models.checkpoint import (
    torch_load_checkpoint, extract_state_dict,
    remove_unwanted_prefixes, filter_matching_tensors,
)


class SegFormerWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        cfg = SegformerConfig.from_pretrained(
            SEGFORMER_BASE_MODEL_ID,
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.base = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_BASE_MODEL_ID,
            config=cfg,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        logits = self.base(pixel_values=x).logits
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def _add_base_prefix(state_dict):
    remapped = OrderedDict()
    for k, v in state_dict.items():
        remapped[k if k.startswith("base.") else f"base.{k}"] = v
    return remapped


def load_segformer(checkpoint_path):
    model = SegFormerWrapper().to(DEVICE)

    ckpt       = torch_load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(ckpt)
    state_dict = remove_unwanted_prefixes(state_dict)
    state_dict = _add_base_prefix(state_dict)

    model_sd = model.state_dict()
    matching, skipped = filter_matching_tensors(state_dict, model_sd)

    if len(matching) == 0:
        raise RuntimeError(
            f"No tensors matched for SegFormer checkpoint.\n"
            f"First 10 keys after remapping:\n"
            + "\n".join(list(state_dict.keys())[:10])
        )

    model_sd.update(matching)
    model.load_state_dict(model_sd, strict=True)

    print(f"[SegFormer] loaded {len(matching)} tensors, skipped {len(skipped)}")
    if skipped:
        for k, reason in skipped[:5]:
            print(f"  skipped  {k}: {reason}")

    return model
