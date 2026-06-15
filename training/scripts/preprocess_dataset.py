#!/usr/bin/env python
"""Preprocess the ATR dataset: merge labels, clean masks, and save to disk."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.preprocessing import run_preprocessing


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess the ATR clothing segmentation dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("ATR_DATASET_ROOT", "./data/atr_merged_preprocessed"),
        help="Directory where the HuggingFace dataset will be saved.",
    )
    parser.add_argument("--hf-dataset-id", type=str, default="ckotait/ATRDataset")
    parser.add_argument("--num-proc", type=int, default=None, help="Parallel workers for HuggingFace map().")
    args = parser.parse_args()

    run_preprocessing(
        output_dir=args.output_dir,
        hf_dataset_id=args.hf_dataset_id,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
