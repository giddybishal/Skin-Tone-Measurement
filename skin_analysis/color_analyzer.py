import logging
import yaml
import numpy as np
from skimage.color import rgb2lab
import os

logger = logging.getLogger(__name__)

class ColorAnalyzer:
    def __init__(self, config_path):
        logger.info(f"Initializing ColorAnalyzer with config: {config_path}")
        self.mapping = self._load_monk_mapping(config_path)

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

    def analyze_region(self, image_rgb, region_mask):
        """
        Analyzes a specific region to get LAB and ITA values.
        
        Args:
            image_rgb: numpy array (H, W, 3) of the cropped face in RGB format.
            region_mask: boolean numpy array (H, W) where True means valid skin.
            
        Returns:
            dict containing L, A, B, ITA, Monk
        """
        if not np.any(region_mask):
            logger.warning("Region mask is empty. Returning None for color analysis.")
            return None

        # Extract only the valid pixels
        valid_pixels_rgb = image_rgb[region_mask] # Shape: (N, 3)
        
        # rgb2lab expects an image of shape (H, W, 3), so we reshape our 1D array of pixels
        # to (N, 1, 3) for conversion.
        pixels_reshaped = valid_pixels_rgb.reshape(-1, 1, 3)
        
        # Convert RGB to LAB (skimage rgb2lab expects RGB in [0, 255] for uint8 or [0, 1] for float)
        # Assuming image_rgb is uint8 [0, 255]
        pixels_lab = rgb2lab(pixels_reshaped) # Shape: (N, 1, 3)
        
        # Calculate means
        L_mean = np.mean(pixels_lab[:, 0, 0])
        a_mean = np.mean(pixels_lab[:, 0, 1])
        b_mean = np.mean(pixels_lab[:, 0, 2])
        
        # Compute ITA
        # Formula: ITA = arctan((L - 50) / b) * (180 / pi)
        # To avoid division by zero:
        b_val = b_mean if b_mean != 0 else 1e-6
        ita = np.arctan((L_mean - 50.0) / b_val) * (180.0 / np.pi)
        
        # Map to Monk scale
        monk_cat = self.get_monk_category(ita)
        
        logger.info(f"Analyzed region -> L:{L_mean:.2f} a:{a_mean:.2f} b:{b_mean:.2f} | ITA:{ita:.2f} | Monk:{monk_cat}")
        
        return {
            'L': L_mean,
            'a': a_mean,
            'b': b_mean,
            'ita': ita,
            'monk': monk_cat
        }
