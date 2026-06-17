import logging
import cv2
import numpy as np
import os
import yaml

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, output_dir, config_path):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.mst_colors = self._load_colors(config_path)

    def _load_colors(self, config_path):
        colors = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                for entry in config['mapping']:
                    # stored as [B, G, R] in config
                    colors[entry['category']] = tuple(entry['color'])
        return colors

    def save_visualizations(self, filename, original_img, bbox, regions, results):
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
