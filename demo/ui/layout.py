import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from config import MODEL_NAMES
from inference import run_inference
from ui.components import legend_html, error_html


_CSS = """
.gradio-container {
    max-width: 1450px !important;
    padding-top: 10px !important;
}
.fit-image {
    height: 52vh !important;
    max-height: 500px !important;
    min-height: 320px !important;
}
.fit-image img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}
.legend-box {
    max-height: 190px;
    overflow-y: auto;
}
"""


def _predict(img, model_name):
    if img is None:
        return None, ""
    try:
        overlay, detected = run_inference(img, model_name)
        return overlay, legend_html(detected)
    except Exception as e:
        return None, error_html(e)


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=_CSS) as demo:
        gr.Markdown("# Clothes Segmentation App")

        model_dropdown = gr.Dropdown(
            choices=MODEL_NAMES,
            value=MODEL_NAMES[0],
            label="Choose Model",
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type="pil",
                    label="Upload Image Here",
                    height=480,
                    elem_classes=["fit-image"],
                )

            with gr.Column(scale=1):
                img_output = gr.Image(
                    type="pil",
                    label="Segmented Image",
                    height=480,
                    elem_classes=["fit-image"],
                )
                legend_out = gr.HTML(
                    label="Detected Classes",
                    elem_classes=["legend-box"],
                )

        segment_btn = gr.Button("Segment", size="lg")
        segment_btn.click(
            fn=_predict,
            inputs=[img_input, model_dropdown],
            outputs=[img_output, legend_out],
        )

    return demo