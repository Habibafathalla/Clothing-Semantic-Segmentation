import sys
from pathlib import Path

# make sure the project root is on sys.path so sub-packages resolve correctly
# regardless of which directory you launch from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEVICE
from models.loader import load_all_models
from ui.layout import build_demo

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    load_all_models()

    demo = build_demo()
    demo.launch()