# 🎨 Skin Tone Region Analysis

An end-to-end, explainable pipeline for analyzing human skin tones across specific facial regions using the **Monk Skin Tone (MST)** scale. Includes an interactive **Gradio** demo for local use, stakeholder presentations, and one-click deployment to **Hugging Face Spaces**.

---

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
uv sync

# 2. Launch the Gradio demo
uv run python app.py
```

Open **http://localhost:7860** in your browser. Upload a face photo and click **Analyze**.

> **Note:** BiSeNet pretrained weights are downloaded automatically on first launch. This requires an internet connection (~100MB download).

---

## 🤗 Hugging Face Spaces Deployment

Deploying to Hugging Face requires **zero additional coding**:

### 1. Create a New Space

- Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- Choose **Gradio** as the SDK
- Select hardware (CPU works, GPU is faster)

### 2. Push the Repository

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

### 3. Wait for Build

The Space will:
1. Install dependencies from `requirements.txt`
2. Download BiSeNet weights automatically
3. Initialize models
4. Launch the Gradio app

**No manual setup scripts, no weight uploads, no environment variables needed.**

---

## 📦 Batch Processing Mode

The original batch pipeline is still available for processing directories of images:

```bash
# 1. Configure settings
# Edit config/pipeline.yaml

# 2. Place images in input_images/
# 3. Run batch analysis
uv run python main.py

# 4. View results
# output/results.csv — metrics for all images
# output/visualizations/ — annotated images per input
```

---

## ⚙️ How the Pipeline Works

```
Face Detection → Face Parsing → Region Extraction → LAB Conversion → ITA Computation → Monk Mapping
```

| Step | Module | Description |
|------|--------|-------------|
| **Detection** | `face_detection/` | MediaPipe Face Mesh (468 landmarks) or MTCNN (5 landmarks) |
| **Parsing** | `face_parsing/` | BiSeNet 19-class semantic segmentation (skin, nose, hair, etc.) |
| **Extraction** | `region_extraction/` | Polygon masks from landmarks ∩ BiSeNet skin mask |
| **Analysis** | `skin_analysis/` | RGB → CIELAB conversion, ITA computation, Monk scale mapping |
| **Visualization** | `visualization/` | Annotated overlays, region masks, landmark debug images |

### Analyzed Regions
- **Forehead** — skin above the eyebrow line
- **Left Cheek** — polygon-bounded area on the left face
- **Right Cheek** — mirror polygon on the right face
- **Nose** — BiSeNet nose class + tight landmark polygon

### ITA Formula

$$ITA = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$$

ITA scores map to the 10-point **Monk Skin Tone** scale via thresholds in `config/monk_mapping.yaml`.

---

## 🏗️ Architecture

```
app.py                          ← Gradio UI (no analysis logic)
  └── services/
        ├── analysis_service.py ← Single-image API: analyze_image(rgb) → results
        └── model_manager.py    ← Load all models once at startup

main.py                         ← Batch processing (existing)

face_detection/                 ← MediaPipe + MTCNN detectors
face_parsing/                   ← BiSeNet semantic parser
region_extraction/              ← Polygon-based region masks
skin_analysis/                  ← LAB/ITA/Monk analysis
visualization/                  ← Render annotated images + masks
config/                         ← Pipeline settings + Monk mapping
examples/                       ← Demo images for Gradio
```

The **service layer** (`services/`) separates pipeline logic from the UI:
- `ModelManager` handles model lifecycle (load once, reuse forever)
- `AnalysisService` is the single entry point for any client (Gradio, API, CLI)
- `app.py` is pure UI wiring — no cv2, no numpy, no analysis logic

---

## 🔧 Configuration

### `config/pipeline.yaml`

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `detector_backend` | `mtcnn`, `mediapipe` | `mediapipe` | Face detector (batch mode) |
| `lab_statistic` | `mean`, `median` | `median` | LAB aggregation method |
| `highlight_removal.enabled` | `true`/`false` | `true` | Remove specular highlights |
| `highlight_removal.percentile` | `1`–`100` | `95` | L-channel brightness cutoff |

### `config/monk_mapping.yaml`

Defines ITA → MST thresholds and BGR colors for each Monk category (MST-1 through MST-10).

---

## 📁 Project Structure

```
├── app.py                      # Gradio demo entry point
├── main.py                     # Batch processing entry point
├── requirements.txt            # HF Spaces dependencies
├── pyproject.toml              # Local uv dependencies
│
├── services/
│   ├── model_manager.py        # Centralized model loading + weight download
│   └── analysis_service.py     # Single-image analysis API
│
├── face_detection/
│   ├── base_detector.py        # Abstract interface
│   ├── mediapipe_detector.py   # MediaPipe Face Mesh (468 landmarks)
│   ├── mtcnn_detector.py       # MTCNN (5 landmarks)
│   └── factory.py              # get_detector() factory
│
├── face_parsing/
│   ├── bisenet_parser.py       # BiSeNet 19-class segmentation
│   ├── bisenet_repo/           # Vendored BiSeNet model code
│   └── setup.py                # Weight download utility
│
├── region_extraction/
│   └── extractor.py            # Region masks from landmarks + skin mask
│
├── skin_analysis/
│   └── color_analyzer.py       # LAB stats, ITA, Monk mapping
│
├── visualization/
│   └── visualizer.py           # Overlays, annotations, debug masks
│
├── config/
│   ├── pipeline.yaml           # Pipeline settings
│   └── monk_mapping.yaml       # ITA → MST thresholds
│
├── examples/                   # Demo images for Gradio
└── logs/                       # Runtime logs
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| **Gradio** | Interactive web UI + HF Spaces deployment |
| **MediaPipe** | 468-point face mesh for polygon-based regions |
| **BiSeNet** (PyTorch) | 19-class semantic face parsing |
| **scikit-image** | RGB → CIELAB conversion |
| **OpenCV** | Image processing, polygon masks |
| **gdown** | Automatic weight download from Google Drive |
