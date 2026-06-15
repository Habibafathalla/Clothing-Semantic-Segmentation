# ATR Clothing Segmentation

The pipeline covers:

1. **Dataset preprocessing** — load ATR from HuggingFace, merge 18 raw labels into 8 classes, clean masks, and save a disk dataset.
2. **Model training** — U-Net, DeepLabV3+, or SegFormer-B0 with shared augmentations, losses, and metrics.
3. **Evaluation** — test-set mIoU, per-class IoU/precision/recall/F1, and confusion matrix plots.

> **Quick start:** see [RUN.md](RUN.md) for step-by-step run instructions (setup, preprocess, train, evaluate).

## Project structure

```
Clothing-Segmentation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/                     
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_visualization.ipynb
│
├── training/                       
│   ├── model/
│   ├── scripts/
│   ├── utils/
│   └── main.py
│
├── demo/                          
│   ├── app.py
│   ├── inference.py
│   └── ui/
│
└── assets/
    └── demo.gif
```
# Cloth Semantic Segmentation App Demo
[Watch the demo video](https://drive.google.com/drive/folders/15K1SkMSZKMn0oZi3znpTwk2gN9rMeQyZ?usp=sharing)

## Installation

Use Python 3.10+ and a virtual environment.

```bash
cd ATR-Clothing-Segmentation
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Install a **PyTorch** build that matches your CUDA driver from [pytorch.org](https://pytorch.org) if the default wheel is not suitable for your GPU.

## Data

The preprocessing script downloads the ATR mirror from HuggingFace:

- Dataset: [`ckotait/ATRDataset`](https://huggingface.co/datasets/ckotait/ATRDataset)

### Preprocess the dataset

```bash
export ATR_DATASET_ROOT="/absolute/path/to/data/atr_merged_preprocessed"

python scripts/preprocess_dataset.py --output-dir "$ATR_DATASET_ROOT"
```

This will:

- Merge the original 18 ATR labels into **8 classes** (Background, Upper-clothes, Skirt, Pants, Dress, Shoes, Human Parts, Accessories)
- Clean masks (remove small blobs, fill holes, morphological closing)
- Save a HuggingFace `DatasetDict` with splits: `train`, `validation`, `test`
- Compute and save inverse-sqrt frequency **class weights** to `class_weights.pt`

Expected split sizes (from the HuggingFace mirror):

| Split       | Samples |
|-------------|---------|
| train       | 16,706  |
| validation  | 1,000   |
| test        | 200     |

Images are resized to **512×384** during training/evaluation.

## Model configuration

Supported backbones (set with `model.name`):

| `model.name`    | Description                          |
|-----------------|--------------------------------------|
| `unet`          | U-Net + ResNet-50 (ImageNet)         |
| `deeplabv3plus` | DeepLabV3+ + ResNet-50 (ImageNet)    |
| `segformer`     | SegFormer-B0 (ADE20K pretrained)     |

Default training hyperparameters (from the notebooks):

| Parameter        | Value   |
|------------------|---------|
| Batch size       | 16      |
| Epochs           | 50      |
| Learning rate    | 6e-5    |
| Optimizer        | AdamW   |
| LR schedule      | Poly decay `(1 - t)^0.9` |
| Loss             | Weighted CE + 0.5 × Dice |
| Early stopping   | Patience 7 on validation mIoU |
| Random seed      | 42      |

## Training

Run from the repository root so imports and Hydra configs resolve.

### U-Net + ResNet-50

```bash
export ATR_DATASET_ROOT="/absolute/path/to/data/atr_merged_preprocessed"

python main.py \
  run.job=train \
  run.name=unet_resnet50 \
  model.name=unet \
  model.encoder_name=resnet50
```

### DeepLabV3+ + ResNet-50

```bash
python main.py \
  run.job=train \
  run.name=deeplab_resnet50 \
  model.name=deeplabv3plus \
  model.encoder_name=resnet50
```

### SegFormer-B0

```bash
python main.py \
  run.job=train \
  run.name=segformer_b0 \
  model.name=segformer
```

Checkpoints, training curves, and saved config are written under:

```
outputs/<run.name>/
├── checkpoints/
│   ├── best_model.pth
│   └── epoch_XX.pth
├── config_saved.yaml
├── history.json
├── training_curves.png
└── test_eval/
    ├── test_metrics.json
    └── confusion_matrix.png
```

### Resume training

```bash
python main.py \
  run.job=train \
  run.name=unet_resnet50 \
  checkpoint.resume=/path/to/checkpoints/epoch_15.pth
```

## Evaluation

Evaluate a saved checkpoint on the test split:

```bash
python main.py \
  run.job=evaluate \
  run.name=unet_resnet50 \
  checkpoint.resume=outputs/unet_resnet50/checkpoints/best_model.pth
```

## Reproducing notebook results

The code is ported from these notebooks:

- `ATR dataset_Preprocessing_Pipeline.ipynb` → `dataset/preprocessing.py`, `scripts/preprocess_dataset.py`
- `u-net-resnet50-final.ipynb` → `model.name=unet`
- `deeplab-resnet50-pipeline.ipynb` → `model.name=deeplabv3plus`
- `segformer-final.ipynb` → `model.name=segformer`

To closely match the notebooks:

1. Preprocess data with `scripts/preprocess_dataset.py`
2. Train with the same model name and default hyperparameters above
3. Use `run.seed=42` (default) for deterministic subset sampling
4. Compare validation mIoU during training and final test metrics in `test_eval/test_metrics.json`

**Note:** Exact metric values may vary slightly across hardware, PyTorch versions, and whether a GPU is used (mixed-precision training is enabled when CUDA is available).

## Environment variables

| Variable           | Purpose                                      |
|--------------------|----------------------------------------------|
| `ATR_DATASET_ROOT` | Path to preprocessed HuggingFace dataset     |
| `OUT_DIR`          | Override default `./outputs` run root          |

## Acknowledgements

- ATR dataset: *Deep Human Parsing with Active Template Regression* (CVPR 2015)
- HuggingFace mirror: [`ckotait/ATRDataset`](https://huggingface.co/datasets/ckotait/ATRDataset)

