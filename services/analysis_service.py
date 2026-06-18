import os
import logging
import time
import cv2
import numpy as np

from visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)

# Resolve project root relative to this file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AnalysisService:
    """
    Single-image analysis API.

    This is the only entry point the Gradio app (or any future API) should use.
    All analysis logic lives here; the UI layer should only call analyze_image().
    """

    def __init__(self, model_manager):
        """
        Args:
            model_manager: An initialized ModelManager instance with all models loaded.
        """
        self.models = model_manager
        self.visualizer = Visualizer.create_for_service(
            config_path=os.path.join(PROJECT_ROOT, 'config', 'monk_mapping.yaml')
        )

    def analyze_image(self, image_rgb):
        """
        Run the full skin tone analysis pipeline on a single image.

        Args:
            image_rgb: numpy array in RGB format (as provided by Gradio).

        Returns:
            dict with keys:
                'annotated_image': RGB numpy array — annotated result with legend
                'overlay_image': RGB numpy array — MST color overlay on skin regions
                'landmark_image': RGB numpy array — landmark dots on original (or None)
                'region_masks': list of (RGB numpy array, label) tuples for gallery
                'metrics': list of dicts for DataFrame display
                'timing': dict with per-step timing in ms
                'error': str or None — error message if analysis failed
        """
        t_start = time.perf_counter()
        timing = {}

        try:
            # Convert RGB (Gradio) → BGR (OpenCV / pipeline internal format)
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # ── Step 1: Face Detection ──
            t0 = time.perf_counter()
            detection_result = self.models.detector.detect_face(img_bgr)
            timing['detect_ms'] = round((time.perf_counter() - t0) * 1000, 1)

            if not detection_result:
                logger.warning("No face detected in uploaded image.")
                return self._error_result(
                    "No suitable face was detected. "
                    "Please upload a clear, front-facing image with good lighting."
                )

            bbox = detection_result['bbox']
            landmarks = detection_result['landmarks']
            x1, y1, x2, y2 = bbox

            # ── Crop face with padding ──
            pad = 20
            h, w = img_bgr.shape[:2]
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(w, x2 + pad)
            py2 = min(h, y2 + pad)

            cropped_face = img_bgr[py1:py2, px1:px2]
            cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

            # ── Step 2: Face Parsing (BiSeNet) ──
            t0 = time.perf_counter()
            parsing_mask = self.models.parser.parse_face(cropped_rgb)
            timing['parse_ms'] = round((time.perf_counter() - t0) * 1000, 1)

            # ── Step 3: Region Extraction ──
            t0 = time.perf_counter()
            landmarks_cropped = landmarks.copy()
            landmarks_cropped[:, 0] -= px1
            landmarks_cropped[:, 1] -= py1

            regions = self.models.extractor.extract_regions(parsing_mask, landmarks_cropped)
            timing['extract_ms'] = round((time.perf_counter() - t0) * 1000, 1)

            # ── Step 4: Color Analysis (LAB / ITA / Monk) ──
            t0 = time.perf_counter()
            results = {}
            for region_name, mask in regions.items():
                res = self.models.color_analyzer.analyze_region(
                    cropped_rgb, mask, region_name
                )
                results[region_name] = res
            timing['analysis_ms'] = round((time.perf_counter() - t0) * 1000, 1)

            # ── Step 5: Render Visualizations (in-memory) ──
            t0 = time.perf_counter()
            padded_bbox = [px1, py1, px2, py2]

            annotated_img = self.visualizer.render_annotated(
                img_bgr, padded_bbox, regions, results,
                backend_name="mediapipe", landmarks=landmarks
            )
            overlay_img = self.visualizer.render_overlay(
                img_bgr, padded_bbox, regions, results
            )
            landmark_img = self.visualizer.render_landmarks(img_bgr, landmarks)
            region_mask_gallery = self.visualizer.render_region_masks(regions)
            timing['vis_ms'] = round((time.perf_counter() - t0) * 1000, 1)

            # ── Build metrics table ──
            metrics_table = self._build_metrics_table(results)

            timing['total_ms'] = round((time.perf_counter() - t_start) * 1000, 1)
            logger.info(
                f"Analysis completed in {timing['total_ms']:.0f}ms "
                f"(detect={timing['detect_ms']}ms, parse={timing['parse_ms']}ms, "
                f"extract={timing['extract_ms']}ms, analysis={timing['analysis_ms']}ms, "
                f"vis={timing['vis_ms']}ms)"
            )

            return {
                'annotated_image': annotated_img,
                'overlay_image': overlay_img,
                'landmark_image': landmark_img,
                'region_masks': region_mask_gallery,
                'metrics': metrics_table,
                'timing': timing,
                'error': None,
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return self._error_result(
                f"Analysis failed due to an internal error. Please try a different image. "
                f"Error: {str(e)}"
            )

    def _build_metrics_table(self, results):
        """Convert per-region results dict into a list-of-dicts for gr.DataFrame."""
        table = []
        # Define consistent display order
        region_order = ['forehead', 'left_cheek', 'right_cheek', 'nose']

        for region_name in region_order:
            res = results.get(region_name)
            if res:
                table.append({
                    'Region': region_name.replace('_', ' ').title(),
                    'L*': round(res['L'], 2),
                    'a*': round(res['a'], 2),
                    'b*': round(res['b'], 2),
                    'ITA': round(res['ita'], 2),
                    'Monk Scale': res['monk'],
                })
            else:
                table.append({
                    'Region': region_name.replace('_', ' ').title(),
                    'L*': '—',
                    'a*': '—',
                    'b*': '—',
                    'ITA': '—',
                    'Monk Scale': '—',
                })

        return table

    @staticmethod
    def _error_result(message):
        """Return a standardized error result dict."""
        return {
            'annotated_image': None,
            'overlay_image': None,
            'landmark_image': None,
            'region_masks': [],
            'metrics': [],
            'timing': {},
            'error': message,
        }
