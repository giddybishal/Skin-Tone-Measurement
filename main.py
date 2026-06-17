import os
import glob
import logging
import cv2
import pandas as pd
from tqdm import tqdm

from face_detection.detector import FaceDetector
from face_parsing.bisenet_parser import FaceParser
from region_extraction.extractor import RegionExtractor
from skin_analysis.color_analyzer import ColorAnalyzer
from visualization.visualizer import Visualizer

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/run.log"),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Starting Skin Tone Region Analysis Pipeline")

    input_dir = 'input_images'
    output_dir = 'output'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize modules
    detector = FaceDetector(use_gpu=False)
    
    # Check if BiSeNet weights are present
    weight_path = os.path.join(os.path.dirname(__file__), 'face_parsing', 'bisenet_repo', 'res', 'cp', '79999_iter.pth')
    if not os.path.exists(weight_path):
        logger.error(f"BiSeNet weights not found at {weight_path}. Run face_parsing/setup.py first.")
        return

    parser = FaceParser(use_gpu=False)
    extractor = RegionExtractor()
    
    config_path = os.path.join('config', 'monk_mapping.yaml')
    color_analyzer = ColorAnalyzer(config_path)
    visualizer = Visualizer(os.path.join(output_dir, 'visualizations'), config_path)

    # Supported image formats
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
    image_paths = list(set(image_paths))

    if not image_paths:
        logger.warning(f"No images found in {input_dir}")
        return

    all_results = []

    for img_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.basename(img_path)
        logger.info(f"\n--- Processing {filename} ---")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            continue
            
        # Step 1: Face Detection
        detection_result = detector.detect_face(img)
        if not detection_result:
            logger.warning(f"Skipping {filename}: No face detected.")
            continue
            
        bbox = detection_result['bbox']
        landmarks = detection_result['landmarks']
        x1, y1, x2, y2 = bbox
        
        # Crop face
        # Add padding if possible
        pad = 20
        h, w = img.shape[:2]
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w, x2 + pad)
        py2 = min(h, y2 + pad)
        
        cropped_face = img[py1:py2, px1:px2]
        
        # Step 2: Face Parsing
        cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        parsing_mask = parser.parse_face(cropped_rgb)
        
        # Save raw parsing debug mask
        visualizer.save_raw_parsing(filename, parsing_mask, [px1, py1, px2, py2], img.shape)
        
        # Step 3: Region Extraction
        # Translate landmarks to cropped face coordinates
        landmarks_cropped = landmarks.copy()
        landmarks_cropped[:, 0] -= px1
        landmarks_cropped[:, 1] -= py1
        
        regions = extractor.extract_regions(parsing_mask, landmarks_cropped)
        
        # Step 4, 5, 6, 7: Color Analysis & ITA & Monk
        results = {}
        row_data = {'filename': filename}
        
        for region_name, mask in regions.items():
            res = color_analyzer.analyze_region(cropped_rgb, mask)
            results[region_name] = res
            
            if res:
                row_data[f'{region_name}_ita'] = res['ita']
                row_data[f'{region_name}_monk'] = res['monk']
                row_data[f'{region_name}_L'] = res['L']
                row_data[f'{region_name}_a'] = res['a']
                row_data[f'{region_name}_b'] = res['b']
            else:
                row_data[f'{region_name}_ita'] = None
                row_data[f'{region_name}_monk'] = None
                row_data[f'{region_name}_L'] = None
                row_data[f'{region_name}_a'] = None
                row_data[f'{region_name}_b'] = None
                
        all_results.append(row_data)
        
        # Visualization
        visualizer.save_visualizations(filename, img, [px1, py1, px2, py2], regions, results)
        logger.info(f"Finished processing {filename}")

    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Processing complete. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
