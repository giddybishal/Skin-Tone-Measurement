import logging
import cv2
import numpy as np
import os
import yaml

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, output_dir=None, config_path=None):
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.mst_colors = {}
        if config_path:
            self.mst_colors = self._load_colors(config_path)

    @classmethod
    def create_for_service(cls, config_path):
        """Create a Visualizer for in-memory rendering (no output directory needed)."""
        instance = cls(output_dir=None, config_path=config_path)
        return instance

    def _load_colors(self, config_path):
        colors = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                for entry in config['mapping']:
                    # stored as [B, G, R] in config
                    colors[entry['category']] = tuple(entry['color'])
        return colors

    def save_visualizations(self, filename, original_img, bbox, regions, results, backend_name="mtcnn", landmarks=None):
        """
        Generates and saves the required visualizations.
        """
        base_name = os.path.splitext(filename)[0]
        
        # Create a dedicated subfolder for this image to prevent clutter
        img_output_dir = os.path.join(self.output_dir, base_name)
        os.makedirs(img_output_dir, exist_ok=True)
        
        logger.info(f"Generating visualizations for {filename} in {img_output_dir}...")
        
        # 1. Save Original Image
        orig_path = os.path.join(img_output_dir, "original.jpg")
        cv2.imwrite(orig_path, original_img)
        
        # 2. Region Overlay Image (using predicted MST colors)
        overlay_img = original_img.copy()
        x1, y1, x2, y2 = bbox
        
        for region_name, mask in regions.items():
            res = results.get(region_name)
            if not res or res['monk'] not in self.mst_colors:
                continue
                
            color = self.mst_colors[res['monk']]
            
            # Map mask back to original image
            full_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=bool)
            
            mh, mw = mask.shape
            ch, cw = y2-y1, x2-x1
            
            if mh != ch or mw != cw:
                mask = cv2.resize(mask.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST).astype(bool)
                
            full_mask[y1:y2, x1:x2] = mask
            
            colored_region = np.zeros_like(original_img)
            colored_region[:] = color
            
            # Blend
            alpha = 0.6
            overlay_img[full_mask] = cv2.addWeighted(overlay_img[full_mask], 1 - alpha, colored_region[full_mask], alpha, 0)
            
        overlay_path = os.path.join(img_output_dir, "overlay.jpg")
        cv2.imwrite(overlay_path, overlay_img)
        
        # 3. Final Annotated Image (Original + Bounding Box + Legend)
        annotated_img = original_img.copy()
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Add legend
        legend_x = 20
        legend_y = 30
        
        # Background for legend
        cv2.rectangle(annotated_img, (10, 10), (350, 20 + 40 * len(results)), (0, 0, 0), -1)
        
        for region_name, res in results.items():
            if not res:
                continue
                
            color = self.mst_colors.get(res['monk'], (255, 255, 255))
            
            # Draw color swatch
            cv2.rectangle(annotated_img, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), color, -1)
            cv2.rectangle(annotated_img, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), (255, 255, 255), 1)
            
            # Draw text
            text = f"{region_name.capitalize()}: {res['monk']} (ITA {res['ita']:.1f})"
            cv2.putText(annotated_img, text, (legend_x + 30, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            legend_y += 40
            
        # Draw detector backend
        cv2.putText(annotated_img, f"Detector: {backend_name.capitalize()}", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Overlay MediaPipe Landmarks if applicable
        if backend_name == 'mediapipe' and landmarks is not None:
            for (x, y) in landmarks:
                cv2.circle(annotated_img, (int(x), int(y)), 1, (0, 255, 0), -1)
            
        annotated_path = os.path.join(img_output_dir, "annotated.jpg")
        cv2.imwrite(annotated_path, annotated_img)
        
        logger.info(f"Saved visualizations for {filename}")

    def save_raw_parsing(self, filename, parsing_mask, bbox, full_shape):
        """
        Saves the raw BiSeNet parsing mask for debugging.
        """
        base_name = os.path.splitext(filename)[0]
        
        x1, y1, x2, y2 = bbox
        mh, mw = parsing_mask.shape
        ch, cw = y2-y1, x2-x1
        
        if mh != ch or mw != cw:
            parsing_mask = cv2.resize(parsing_mask.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
            
        vis_mask = (parsing_mask * (255 // 18)).astype(np.uint8)
        colored_mask = cv2.applyColorMap(vis_mask, cv2.COLORMAP_JET)
        
        full_debug = np.zeros((full_shape[0], full_shape[1], 3), dtype=np.uint8)
        full_debug[y1:y2, x1:x2] = colored_mask
        
        debug_dir = os.path.join(os.path.dirname(self.output_dir), 'debug_masks')
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_path = os.path.join(debug_dir, f"{base_name}_parsing.jpg")
        cv2.imwrite(debug_path, full_debug)

    def save_landmarks_debug(self, filename, image, landmarks):
        """
        Saves a debug image with landmarks overlaid.
        """
        if landmarks is None:
            return
            
        base_name = os.path.splitext(filename)[0]
        debug_dir = os.path.join(self.output_dir, '..', 'debug_landmarks')
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_img = image.copy()
        for (x, y) in landmarks:
            cv2.circle(debug_img, (int(x), int(y)), 1, (0, 255, 0), -1)
            
        debug_path = os.path.normpath(os.path.join(debug_dir, f"{base_name}_landmarks.jpg"))
        cv2.imwrite(debug_path, debug_img)

    def save_region_masks(self, filename, regions):
        """
        Saves binary region masks to the debug_masks folder.
        """
        base_name = os.path.splitext(filename)[0]
        debug_dir = os.path.join(self.output_dir, '..', 'debug_masks')
        os.makedirs(debug_dir, exist_ok=True)
        
        for region_name, mask in regions.items():
            # Convert boolean mask to uint8 255
            bin_mask = (mask.astype(np.uint8) * 255)
            mask_path = os.path.normpath(os.path.join(debug_dir, f"{base_name}_{region_name}_mask.png"))
            cv2.imwrite(mask_path, bin_mask)

    # ─── In-Memory Render Methods (for Gradio / API use) ─────────────────

    def render_annotated(self, original_img_bgr, bbox, regions, results,
                         backend_name="mediapipe", landmarks=None):
        """
        Render annotated image with bounding box, legend, and optional landmarks.

        Returns:
            RGB numpy array suitable for Gradio display.
        """
        x1, y1, x2, y2 = bbox
        annotated = original_img_bgr.copy()

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Legend background
        cv2.rectangle(annotated, (10, 10), (350, 20 + 40 * len(results)), (0, 0, 0), -1)

        legend_x, legend_y = 20, 30
        for region_name, res in results.items():
            if not res:
                continue
            color = self.mst_colors.get(res['monk'], (255, 255, 255))
            cv2.rectangle(annotated, (legend_x, legend_y - 15),
                          (legend_x + 20, legend_y + 5), color, -1)
            cv2.rectangle(annotated, (legend_x, legend_y - 15),
                          (legend_x + 20, legend_y + 5), (255, 255, 255), 1)
            text = f"{region_name.replace('_', ' ').title()}: {res['monk']} (ITA {res['ita']:.1f})"
            cv2.putText(annotated, text, (legend_x + 30, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            legend_y += 40

        cv2.putText(annotated, f"Detector: {backend_name.capitalize()}",
                    (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Landmarks overlay
        if landmarks is not None:
            for (x, y) in landmarks:
                cv2.circle(annotated, (int(x), int(y)), 1, (0, 255, 0), -1)

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def render_overlay(self, original_img_bgr, bbox, regions, results):
        """
        Render region overlay image with MST colors blended onto skin regions.

        Returns:
            RGB numpy array suitable for Gradio display.
        """
        x1, y1, x2, y2 = bbox
        overlay = original_img_bgr.copy()

        for region_name, mask in regions.items():
            res = results.get(region_name)
            if not res or res['monk'] not in self.mst_colors:
                continue

            color = self.mst_colors[res['monk']]
            full_mask = np.zeros((original_img_bgr.shape[0], original_img_bgr.shape[1]), dtype=bool)

            mh, mw = mask.shape
            ch, cw = y2 - y1, x2 - x1

            if mh != ch or mw != cw:
                mask = cv2.resize(mask.astype(np.uint8), (cw, ch),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            full_mask[y1:y2, x1:x2] = mask

            colored = np.zeros_like(original_img_bgr)
            colored[:] = color

            alpha = 0.6
            overlay[full_mask] = cv2.addWeighted(
                overlay[full_mask], 1 - alpha, colored[full_mask], alpha, 0
            )

        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    def render_landmarks(self, image_bgr, landmarks):
        """
        Render landmark debug image.

        Returns:
            RGB numpy array suitable for Gradio display, or None if no landmarks.
        """
        if landmarks is None:
            return None

        debug = image_bgr.copy()
        for (x, y) in landmarks:
            cv2.circle(debug, (int(x), int(y)), 1, (0, 255, 0), -1)

        return cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)

    def render_region_masks(self, regions):
        """
        Render region masks as a list of (image, label) tuples for gr.Gallery.

        Returns:
            List of (RGB numpy array, region_name) tuples.
        """
        gallery_items = []
        for region_name, mask in regions.items():
            # Convert bool mask to visible grayscale → RGB
            vis = (mask.astype(np.uint8) * 255)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
            label = region_name.replace('_', ' ').title()
            gallery_items.append((vis_rgb, label))
        return gallery_items
