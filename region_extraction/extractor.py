import logging
import numpy as np

logger = logging.getLogger(__name__)

class RegionExtractor:
    def __init__(self):
        logger.info("Initializing RegionExtractor")
        
        # BiSeNet classes
        self.class_skin = 1
        self.class_nose = 10
        
        # We also might want to explicitly exclude some classes just to be safe, 
        # though taking only class_skin inherently excludes them.
        self.valid_skin_classes = [self.class_skin]

    def extract_regions(self, parsing_mask, landmarks_cropped):
        """
        Extracts specific facial regions using geometric heuristics on the skin mask.
        
        Args:
            parsing_mask: numpy array (H, W) of class labels from BiSeNet.
            landmarks_cropped: numpy array (5, 2) of landmarks in the cropped mask coordinate system.
                Indices: 0: left_eye, 1: right_eye, 2: nose, 3: left_mouth, 4: right_mouth
                
        Returns:
            dict mapping region names to boolean numpy arrays (H, W)
        """
        logger.info("Extracting regions based on parsing mask and landmarks...")
        
        H, W = parsing_mask.shape
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        
        # Base skin mask
        skin_mask = (parsing_mask == self.class_skin)
        
        # Unpack landmarks (x, y)
        left_eye = landmarks_cropped[0]
        right_eye = landmarks_cropped[1]
        nose = landmarks_cropped[2]
        left_mouth = landmarks_cropped[3]
        right_mouth = landmarks_cropped[4]
        
        # We need to account for the fact that InsightFace left/right are from the subject's perspective
        # Subject's left eye is on the right side of the image usually, but let's just use x-coordinates
        # to determine image-left and image-right.
        img_left_eye = left_eye if left_eye[0] < right_eye[0] else right_eye
        img_right_eye = right_eye if left_eye[0] < right_eye[0] else left_eye
        
        eye_y_min = min(left_eye[1], right_eye[1])
        eye_y_max = max(left_eye[1], right_eye[1])
        mouth_y_max = max(left_mouth[1], right_mouth[1])
        mouth_y_min = min(left_mouth[1], right_mouth[1])
        
        nose_x = nose[0]
        nose_y = nose[1]

        regions = {}
        
        # 1. Nose: Use BiSeNet's nose class
        regions['nose'] = (parsing_mask == self.class_nose)
        
        # 2. Forehead: Skin above the eyes
        # We add a small buffer above the eye to avoid the eyebrow area just in case
        forehead_mask = skin_mask & (y_coords < eye_y_min - 5)
        regions['forehead'] = forehead_mask
        
        # 3. Left Cheek (Image Left): Skin below eyes, above mouth, left of nose
        left_cheek_mask = skin_mask & (y_coords > eye_y_max + 5) & (y_coords < mouth_y_max) & (x_coords < nose_x)
        regions['left_cheek'] = left_cheek_mask
        
        # 4. Right Cheek (Image Right): Skin below eyes, above mouth, right of nose
        right_cheek_mask = skin_mask & (y_coords > eye_y_max + 5) & (y_coords < mouth_y_max) & (x_coords > nose_x)
        regions['right_cheek'] = right_cheek_mask
        
        # 5. Jaw: Skin below the mouth
        jaw_mask = skin_mask & (y_coords > mouth_y_max + 15)
        regions['jaw'] = jaw_mask
        
        for name, mask in regions.items():
            pixel_count = np.sum(mask)
            logger.info(f"Region '{name}' skin pixels: {pixel_count}")
            
        return regions
