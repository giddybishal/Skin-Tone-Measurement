# 🎨 Skin Tone Region Analysis Pipeline (V2)

An end-to-end, explainable pipeline for analyzing human skin tones across specific facial regions using the **Monk Skin Tone (MST)** scale.

Instead of relying on black-box monoliths, this prototype uses a modular architecture—ensuring that every decision, segmentation boundary, and color mapping is completely transparent and mathematically verifiable.

---

## 🛠️ Tech Stack

| Tool / Library | Purpose | Why We Chose It |
| :--- | :--- | :--- |
| **`uv`** | Dependency Management | Lightning-fast, reproducible, strictly version-locked environments. |
| **`facenet-pytorch`** | Face Detection (MTCNN) | Robust MTCNN architecture for bounding box + 5 key landmarks. |
| **`mediapipe`** | Face Detection (Face Mesh) | Dense 468-point face mesh for precise polygon-based region extraction. |
| **`BiSeNet`** (PyTorch) | Semantic Face Parsing | Pixel-perfect 19-class segmentation of facial components (skin, nose, hair, etc.). |
| **`scikit-image`** | Color Space Conversion | Accurate RGB → CIELAB conversion for ITA computation. |
| **`OpenCV` (cv2)** | Image I/O & Visualization | Reading/writing images, generating overlays, polygon mask creation. |

---

## ⚙️ How the Pipeline Works

```
Face Detection → Face Parsing → Region Extraction → Skin Pixel Extraction → LAB Conversion → ITA Computation → Monk Mapping
```

### 1. Face & Landmark Detection (`face_detection/`)

The image is processed by the configured detector backend:

- **MTCNN** (`facenet-pytorch`): Returns a bounding box and 5 landmarks (left eye, right eye, nose, left mouth, right mouth).
- **MediaPipe Face Mesh**: Returns a bounding box and 468 dense facial landmarks for precise polygon-based region extraction.

Both backends implement the same `FaceDetectorBase` interface, so switching requires only a config change.

**Files:**
- `base_detector.py` — Abstract interface
- `mtcnn_detector.py` — MTCNN implementation
- `mediapipe_detector.py` — MediaPipe Face Mesh implementation
- `factory.py` — `get_detector()` factory function

### 2. Semantic Parsing (`face_parsing/`)

The cropped face is sent to the pre-trained **BiSeNet** model. BiSeNet evaluates every pixel and assigns it to one of 19 classes (e.g., Background, Skin, Left Brow, Right Eye, Hair, Cloth). We isolate the `skin` class (1) and the `nose` class (10).

### 3. Region Extraction (`region_extraction/`)

BiSeNet tells us *what* is skin, but not *where* on the face. We combine landmarks with the skin mask to carve out localized zones:

**MTCNN strategy** (5 landmarks → geometric heuristics):
- **Forehead:** Skin pixels above the eye line (with a 5px buffer below the eyebrows).
- **Cheeks (L/R):** Skin between the eye and mouth lines, split vertically by the nose x-coordinate.
- **Jaw:** Skin below the mouth (offset 15px to avoid upper chin).
- **Nose:** Directly from BiSeNet's dedicated nose class.

**MediaPipe strategy** (468 landmarks → polygon masks):
- **Forehead:** Skin above the minimum eyebrow Y-coordinate (landmarks 70, 63, 105, 66, 107, 336, 296, 334, 293, 300).
- **Left Cheek:** Polygon from landmarks `[123, 147, 192, 214, 207, 216, 206, 205, 36, 50, 118, 119, 100]` intersected with skin mask.
- **Right Cheek:** Polygon from landmarks `[352, 376, 416, 434, 427, 436, 426, 425, 266, 280, 347, 348, 329]` intersected with skin mask.
- **Nose:** BiSeNet class 10 combined with a polygon from landmarks `[168, 197, 5, 4, 1, 19, 94, 2, 275, 440, 274, 45, 220, 215]`.
- **Jaw:** Polygon following the chin contour and lower lip from landmarks `[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, ...]` intersected with skin mask.

All polygon masks are created with `cv2.fillPoly` and intersected with the BiSeNet skin mask to ensure only true skin pixels are measured.

### 4. Color Analysis & ITA Calculation (`skin_analysis/`)

For each region, valid skin pixels go through:

1. **Highlight Removal** (optional): Removes the brightest N% of pixels (by L-channel) to eliminate specular reflections on forehead and nose.
2. **Beard Filtering** (optional, jaw only): Removes the darkest N% of pixels (by L-channel) to exclude facial hair.
3. **LAB Conversion**: Remaining RGB pixels are converted to the **L\*a\*b\*** color space.
4. **Statistic Computation**: Either `median` (default, more robust to outliers) or `mean` of L\*, a\*, b\* values.
5. **ITA Calculation**:
   > $ITA = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$

### 5. Monk Scale Mapping & Visualization (`visualization/`)

The ITA score maps to the 10-point **Monk Skin Tone (MST)** scale using thresholds in `config/monk_mapping.yaml`.

**Output files per image:**
- `original.jpg` — Untouched input
- `overlay.jpg` — Face tinted with the predicted MST color per region
- `annotated.jpg` — Clean legend with bounding box, detector name, and MST labels

**Debug outputs:**
- `output/debug_masks/` — Binary `.png` masks for each region (forehead, left_cheek, right_cheek, nose, jaw)
- `output/debug_landmarks/` — Landmark overlay images (MediaPipe backend only)
- `output/debug_masks/` — Raw BiSeNet parsing visualization

### 6. Inference Timing

Every pipeline step is individually timed per image:

| Metric | Description |
| :--- | :--- |
| `time_detect_ms` | Face detection (MTCNN or MediaPipe) |
| `time_parse_ms` | BiSeNet face parsing |
| `time_extract_ms` | Region extraction from landmarks |
| `time_analysis_ms` | LAB conversion, filtering, ITA, Monk mapping |
| `time_vis_ms` | Saving visualizations |
| `time_total_ms` | End-to-end for that image |

Timing data is logged and saved as columns in `output/results.csv`.

---

## 🚀 Getting Started

### 1. Setup Your Environment
Ensure you have `uv` installed, then synchronize the environment and install dependencies.
```bash
uv sync
```

### 2. Download the Model Weights
Download the pre-trained BiSeNet weights (this only needs to be done once).
```bash
uv run python face_parsing/setup.py
```

### 3. Configure the Pipeline
Edit `config/pipeline.yaml` to choose your settings:
```yaml
detector_backend: mediapipe  # mtcnn or mediapipe
lab_statistic: median        # median or mean

highlight_removal:
  enabled: true
  percentile: 95   # Remove top 5% brightest pixels

beard_filter:
  enabled: true
  percentile: 10   # Remove bottom 10% darkest pixels (jaw only)
```

### 4. Run an Analysis
Place `.jpg`, `.png`, or `.webp` files in `input_images/`, then:
```bash
uv run python main.py
```

### 5. View Results
- **Visual Proof**: `output/visualizations/<image_name>/` — region overlays, legends, and annotations.
- **Raw Data**: `output/results.csv` — L\*, a\*, b\*, ITA, Monk category, timing, detector backend, and statistic type for every region.
- **Debug Masks**: `output/debug_masks/` — binary masks proving exactly which pixels were used.
- **Debug Landmarks**: `output/debug_landmarks/` — landmark visualizations (MediaPipe only).
- **Logs**: `logs/run.log` — full execution trace including pixel counts before/after filtering.

---

## 🔧 Configuration

### `config/pipeline.yaml`

| Key | Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `detector_backend` | `mtcnn`, `mediapipe` | `mtcnn` | Which face detector to use |
| `lab_statistic` | `mean`, `median` | `median` | How to aggregate LAB values across skin pixels |
| `highlight_removal.enabled` | `true`, `false` | `true` | Remove specular highlights before analysis |
| `highlight_removal.percentile` | `1`–`100` | `95` | Brightness cutoff (pixels above this L-channel percentile are removed) |
| `beard_filter.enabled` | `true`, `false` | `true` | Remove dark pixels from jaw region |
| `beard_filter.percentile` | `1`–`100` | `10` | Darkness cutoff (pixels below this L-channel percentile are removed) |

### `config/monk_mapping.yaml`

Defines the ITA → MST mapping thresholds and BGR colors for each Monk category. Edit to adjust classification boundaries.

---

## 📁 Project Structure

```
first_prototype/
├── config/
│   ├── monk_mapping.yaml       # ITA → MST thresholds & colors
│   └── pipeline.yaml           # Detector, statistic, and filter settings
├── face_detection/
│   ├── base_detector.py        # Abstract FaceDetectorBase interface
│   ├── mtcnn_detector.py       # MTCNN implementation
│   ├── mediapipe_detector.py   # MediaPipe Face Mesh implementation
│   └── factory.py              # get_detector() factory
├── face_parsing/
│   └── bisenet_parser.py       # BiSeNet 19-class face segmentation
├── region_extraction/
│   └── extractor.py            # Region masks from landmarks + skin mask
├── skin_analysis/
│   └── color_analyzer.py       # LAB stats, filtering, ITA, Monk mapping
├── visualization/
│   └── visualizer.py           # Overlays, annotations, debug masks/landmarks
├── main.py                     # Pipeline orchestrator with timing
├── pyproject.toml              # Dependencies (uv)
└── README.md
```

---

## 📊 MTCNN vs MediaPipe: Comparison Notes

| Aspect | MTCNN | MediaPipe |
| :--- | :--- | :--- |
| **Landmarks** | 5 points (eyes, nose, mouth) | 468 points (full face mesh) |
| **Region Strategy** | Geometric heuristics (line splits) | Polygon masks (anatomically precise) |
| **Speed** | Moderate (~200–500ms/image) | Fast (~10–30ms/image) |
| **Forehead Coverage** | Relies on skin mask above eye line | Uses eyebrow landmarks for precise cutoff |
| **Cheek Precision** | Simple nose-x split | Polygon contours following actual cheek anatomy |
| **Dependencies** | `facenet-pytorch`, `torch` | `mediapipe` (lighter weight) |
| **Best For** | Compatibility, established workflows | Precision, speed, dense landmark overlays |

> **Recommendation**: Use **MediaPipe** for new analyses—it provides more anatomically precise regions and faster inference. Use **MTCNN** for backward compatibility with earlier results.
