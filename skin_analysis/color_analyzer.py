import logging
import yaml
import numpy as np
from skimage.color import rgb2lab
import os

logger = logging.getLogger(__name__)

class ColorAnalyzer:
    def __init__(self, monk_mapping_path, pipeline_config=None):
        logger.info(f"Initializing ColorAnalyzer")
        self.mapping = self._load_monk_mapping(monk_mapping_path)
        self.config = pipeline_config or {}
        
        self.statistic = self.config.get('lab_statistic', 'median')
        self.highlight_cfg = self.config.get('highlight_removal', {'enabled': False})
        self.beard_cfg = self.config.get('beard_filter', {'enabled': False})
        
        logger.info(f"LAB Statistic: {self.statistic}")

    def _load_monk_mapping(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config['mapping']

    def get_monk_category(self, ita_value):
        for entry in self.mapping:
            if entry['min_ita'] < ita_value <= entry['max_ita']:
                return entry['category']
        return "Unknown"

    def analyze_region(self, image_rgb, region_mask, region_name="unknown"):
        """
        Analyzes a specific region to get LAB and ITA values.
        """
        if not np.any(region_mask):
            logger.warning(f"Region mask {region_name} is empty. Returning None.")
            return None

        # Extract only the valid pixels
        valid_pixels_rgb = image_rgb[region_mask] # Shape: (N, 3)
        orig_pixel_count = len(valid_pixels_rgb)
        
        pixels_reshaped = valid_pixels_rgb.reshape(-1, 1, 3)
        pixels_lab = rgb2lab(pixels_reshaped) # Shape: (N, 1, 3)
        
        L_vals = pixels_lab[:, 0, 0]
        a_vals = pixels_lab[:, 0, 1]
        b_vals = pixels_lab[:, 0, 2]

        # Apply filtering
        valid_indices = np.ones(len(L_vals), dtype=bool)

        # Highlight Removal
        if self.highlight_cfg.get('enabled', False):
            pct = self.highlight_cfg.get('percentile', 95)
            threshold = np.percentile(L_vals, pct)
            # keep pixels darker than the threshold
            valid_indices &= (L_vals <= threshold)


        # Apply indices
        L_vals = L_vals[valid_indices]
        a_vals = a_vals[valid_indices]
        b_vals = b_vals[valid_indices]
        
        filtered_pixel_count = len(L_vals)
        if filtered_pixel_count == 0:
            logger.warning(f"Region {region_name} has no pixels left after filtering.")
            return None

        if orig_pixel_count != filtered_pixel_count:
            logger.info(f"{region_name.capitalize()}: Original pixels: {orig_pixel_count}, After filtering: {filtered_pixel_count}")

        # Calculate stats
        if self.statistic == 'mean':
            L_stat = np.mean(L_vals)
            a_stat = np.mean(a_vals)
            b_stat = np.mean(b_vals)
        else:
            L_stat = np.median(L_vals)
            a_stat = np.median(a_vals)
            b_stat = np.median(b_vals)
        
        # Compute ITA
        b_val = b_stat if b_stat != 0 else 1e-6
        ita = np.arctan((L_stat - 50.0) / b_val) * (180.0 / np.pi)
        
        # Map to Monk scale
        monk_cat = self.get_monk_category(ita)
        
        logger.info(f"{region_name.capitalize()} {self.statistic.capitalize()} LAB: L={L_stat:.2f} a={a_stat:.2f} b={b_stat:.2f} | ITA={ita:.2f} | Monk={monk_cat}")
        
        return {
            'L': L_stat,
            'a': a_stat,
            'b': b_stat,
            'ita': ita,
            'monk': monk_cat
        }
