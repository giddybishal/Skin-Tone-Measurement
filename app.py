"""
Skin Tone Region Analysis — Gradio Application

This is the main entry point for the interactive demo.
Run with: uv run python app.py
"""

import os
import logging
import time

import gradio as gr
import pandas as pd

# ── Logging Setup ────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("app")

# ── Model Initialization (runs once at startup) ─────────────────────────────
logger.info("Starting Skin Tone Region Analysis application...")

from services.model_manager import ModelManager
from services.analysis_service import AnalysisService

model_manager = ModelManager()
model_manager.load_models()

service = AnalysisService(model_manager)

logger.info("Application ready.")


# ── Gradio Callback ─────────────────────────────────────────────────────────

def analyze(image):
    """
    Main Gradio callback. Receives an RGB numpy array from gr.Image,
    runs analysis, and returns all outputs.
    """
    if image is None:
        gr.Warning("Please upload an image first.")
        return None, None, None, None, None, ""

    result = service.analyze_image(image)

    if result['error']:
        gr.Warning(result['error'])
        return None, None, None, None, None, ""

    # Build DataFrame for metrics table
    metrics_df = pd.DataFrame(result['metrics'])

    # Build timing markdown
    t = result['timing']
    timing_md = (
        f"⏱️ **Total: {t.get('total_ms', '?')}ms** — "
        f"Detection: {t.get('detect_ms', '?')}ms · "
        f"Parsing: {t.get('parse_ms', '?')}ms · "
        f"Extraction: {t.get('extract_ms', '?')}ms · "
        f"Analysis: {t.get('analysis_ms', '?')}ms · "
        f"Visualization: {t.get('vis_ms', '?')}ms"
    )

    return (
        result['annotated_image'],
        metrics_df,
        result['overlay_image'],
        result['landmark_image'],
        result['region_masks'],
        timing_md,
    )


# ── Gradio UI ────────────────────────────────────────────────────────────────

css = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}
h1 {
    text-align: center;
    margin-bottom: 0.2em !important;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 1.5em;
    font-size: 1.05em;
}
"""

# Collect example images
examples_dir = os.path.join(os.path.dirname(__file__), "examples")
example_images = []
if os.path.isdir(examples_dir):
    for fname in sorted(os.listdir(examples_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            example_images.append(os.path.join(examples_dir, fname))


gradio_theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="stone",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="Skin Tone Region Analysis") as demo:

    # ── Header ──
    gr.Markdown("# 🎨 Skin Tone Region Analysis")
    gr.Markdown(
        "Upload a face photo to analyze skin tone across facial regions using the "
        "**Monk Skin Tone (MST)** scale. The pipeline uses MediaPipe for landmark detection, "
        "BiSeNet for semantic parsing, and CIELAB color analysis.",
        elem_classes="subtitle",
    )

    with gr.Row():
        # ── Left Column: Input ──
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="numpy",
                label="Upload Image",
                sources=["upload", "clipboard"],
                height=400,
            )
            analyze_btn = gr.Button("🔍 Analyze Skin Tone", variant="primary", size="lg")

        # ── Right Column: Main Output ──
        with gr.Column(scale=1):
            annotated_output = gr.Image(
                label="Annotated Result",
                interactive=False,
                height=400,
            )

    # ── Metrics Table ──
    gr.Markdown("### 📊 Region Metrics")
    metrics_table = gr.DataFrame(
        label="Skin Tone Metrics per Region",
        interactive=False,
        wrap=True,
    )

    # ── Timing ──
    timing_output = gr.Markdown("")

    # ── Debug Accordion ──
    with gr.Accordion("🔬 Advanced Diagnostics", open=False):
        gr.Markdown(
            "Detailed pipeline outputs for technical review. "
            "Useful for verifying region masks, landmark placement, and color overlays."
        )
        with gr.Row():
            overlay_output = gr.Image(label="MST Color Overlay", interactive=False)
            landmark_output = gr.Image(label="Landmark Visualization", interactive=False)

        gr.Markdown("#### Region Masks")
        masks_gallery = gr.Gallery(
            label="Individual Region Masks",
            columns=4,
            object_fit="contain",
            height=250,
        )

    # ── Examples ──
    if example_images:
        gr.Markdown("### 📷 Try an Example")
        gr.Examples(
            examples=example_images,
            inputs=input_image,
            label="Click an image to load it",
        )

    # ── Wire up the callback ──
    analyze_btn.click(
        fn=analyze,
        inputs=[input_image],
        outputs=[
            annotated_output,
            metrics_table,
            overlay_output,
            landmark_output,
            masks_gallery,
            timing_output,
        ],
    )

    # Also trigger on image upload
    input_image.change(
        fn=analyze,
        inputs=[input_image],
        outputs=[
            annotated_output,
            metrics_table,
            overlay_output,
            landmark_output,
            masks_gallery,
            timing_output,
        ],
    )

# ── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gradio_theme, css=css)
