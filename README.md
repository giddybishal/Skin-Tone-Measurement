# 🎨 Skin Tone Region Analysis Prototype

Welcome to the **Skin Tone Region Analysis Prototype**! This project provides an end-to-end, explainable pipeline for analyzing human skin tones across specific facial regions using the Monk Skin Tone (MST) scale. 

Instead of relying on black-box monoliths, this prototype uses a modular architecture—ensuring that every decision, segmentation boundary, and color mapping is completely transparent and mathematically verifiable.

---

## 🛠️ The Tech Stack: What & Why

We carefully selected specific tools and libraries to prioritize **accuracy**, **debuggability**, and **transparency**.

| Tool / Library | Purpose in Pipeline | Why We Chose It |
| :--- | :--- | :--- |
| **`uv`** | Dependency Management | It ensures lightning-fast, reproducible, and strictly version-locked environment setups. |
| **`facenet-pytorch`** | Face Detection & Landmarks | Provides the highly robust **MTCNN** architecture. It reliably detects the face bounding box and 5 critical landmarks (eyes, nose, mouth) without the heavy build-tool requirements of other libraries like `dlib` or `tensorflow`. |
| **`BiSeNet`** (PyTorch) | Semantic Face Parsing | We use a dedicated PyTorch implementation of BiSeNet (19-classes) to get pixel-perfect masks of skin, nose, eyes, and hair. It's an industry standard for facial component segmentation. |
| **`scikit-image`** | Color Space Conversion | Used to convert standard RGB pixels into the **CIELAB (L*a*b*)** color space, which is required to accurately compute the Individual Typology Angle (ITA). |
| **`OpenCV` (cv2)** | Image Processing & Visualization | The backbone for reading/writing images and generating the intuitive colored overlays and annotations in our output. |

---

## ⚙️ How the Pipeline Works

When you run an image through this prototype, it goes through a strict 5-step process:

### 1. Face & Landmark Detection (`face_detection/`)
The image is passed to `MTCNN`. The detector finds the primary face and extracts its **bounding box** along with the precise coordinates of the left eye, right eye, nose, left mouth, and right mouth.

### 2. Semantic Parsing (`face_parsing/`)
The cropped face is sent to the pre-trained **BiSeNet** model. BiSeNet evaluates every pixel and assigns it to one of 19 classes (e.g., Background, Skin, Left Brow, Right Eye, Hair, Cloth). We isolate the `skin` class and the `nose` class.

### 3. Geometric Region Extraction (`region_extraction/`)
BiSeNet tells us what is "skin", but it doesn't separate the cheeks from the forehead. We use the 5 facial landmarks to mathematically carve the skin mask into localized zones:
*   **Forehead:** Skin pixels located physically above the eyes.
*   **Jaw:** Skin pixels located below the mouth (offset by 15 pixels to ensure we don't catch the upper chin).
*   **Cheeks (Left/Right):** Skin pixels between the eyes and the mouth, split vertically by the nose.
*   **Nose:** Directly mapped from BiSeNet's dedicated nose class.

### 4. Color Analysis & ITA Calculation (`skin_analysis/`)
For each defined region, we filter out all non-skin pixels (like hair or eyebrows). The remaining valid RGB pixels are converted to the **L*a*b*** color space. 
We then calculate the **Individual Typology Angle (ITA)** using the formula:
> $ITA = \arctan(\frac{L^* - 50}{b^*}) \times (\frac{180}{\pi})$

### 5. Monk Scale Mapping & Visualization (`visualization/`)
The computed ITA score is mapped to the 10-point **Monk Skin Tone (MST)** scale using thresholds defined in `config/monk_mapping.yaml`. 
Finally, the system generates visual proof:
*   `original.jpg`: The untouched input.
*   `overlay.jpg`: The face tinted with the actual predicted MST color for each respective region.
*   `annotated.jpg`: A clean legend mapped to the bounding box for rapid human verification.

---

## 🚀 Getting Started

### 1. Setup Your Environment
Ensure you have `uv` installed, then synchronize the environment and install dependencies.
```bash
# This creates a virtual environment (.venv) using Python 3.11 and installs requirements.
uv sync
```

### 2. Download the Model Weights
We need to download the pre-trained BiSeNet weights (this only needs to be done once).
```bash
uv run python face_parsing/setup.py
```

### 3. Run an Analysis
Place any `.jpg`, `.png`, or `.webp` files you want to analyze into the `input_images/` directory. Then execute the orchestrator:
```bash
uv run python main.py
```

### 4. View Results
*   **Visual Proof**: Open `output/visualizations/<image_name>/` to see the region overlays and legends.
*   **Raw Data**: Open `output/results.csv` to see the exact numerical values (L*, a*, b*, ITA, and Monk category) for every region across all processed images.
*   **Logs**: Check `logs/run.log` for a play-by-play execution trace.

---

## 🔧 Configuration

Want to adjust the ITA boundaries for the Monk Scale? Simply edit `config/monk_mapping.yaml`. The system will automatically use your new thresholds and colors on the next run!
