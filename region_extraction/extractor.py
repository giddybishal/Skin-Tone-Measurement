import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class RegionExtractor:
    def __init__(self):
        logger.info("Initializing RegionExtractor")
        
        # BiSeNet classes
        self.class_skin = 1
        self.class_nose = 10
        
        self.valid_skin_classes = [self.class_skin]

        # Anatomical reasoning for MediaPipe landmarks:
        # These are carefully chosen points that bound the major facial areas.
        # Intersecting these polygons with the BiSeNet skin mask ensures we only get skin.
        self.mp_indices = {
            # Forehead: Bounded by the top face contour and eyebrows.
            # Using a simplified bounding box based on topmost points and eyebrows is more robust 
            # than a strict polygon which might miss skin up to the hairline.
            # Left Cheek: Face contour (cheekbone level) → below eye → nose side → mouth level.
            'left_cheek_poly': [162, 127, 234, 147, 213, 192, 214, 207, 216, 206, 205, 36, 50, 118, 119, 100, 101],
            # Right Cheek: Mirror of left cheek polygon.
            'right_cheek_poly': [389, 356, 454, 376, 433, 416, 434, 427, 436, 426, 425, 266, 280, 347, 348, 329, 330],
            # Nose: Tight polygon along the ridge, tip, and nasal alae to prevent side bleeding.
            'nose_poly': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 274, 45, 220, 115, 4, 275, 440, 168],
        }

    def extract_regions(self, parsing_mask, landmarks_cropped):
        """
        Extracts specific facial regions.
        Automatically chooses strategy based on number of landmarks.
        """
        if len(landmarks_cropped) > 100:
            return self._extract_mediapipe(parsing_mask, landmarks_cropped)
        else:
            return self._extract_mtcnn(parsing_mask, landmarks_cropped)

    def _extract_mediapipe(self, parsing_mask, landmarks_cropped):
        logger.info("Extracting regions based on parsing mask and MediaPipe landmarks...")
        H, W = parsing_mask.shape
        skin_mask = (parsing_mask == self.class_skin)
        regions = {}

        # 1. Nose: Combine BiSeNet nose mask + polygon intersection for safety
        nose_bisenet = (parsing_mask == self.class_nose)
        nose_poly_mask = self._create_poly_mask((H, W), landmarks_cropped, self.mp_indices['nose_poly'])
        regions['nose'] = nose_bisenet | (skin_mask & nose_poly_mask)

        # 2. Forehead: Skin above the eyes
        # Find the highest eyebrow points
        left_eyebrow_y = np.min(landmarks_cropped[[70, 63, 105, 66, 107], 1])
        right_eyebrow_y = np.min(landmarks_cropped[[336, 296, 334, 293, 300], 1])
        eye_y_min = min(left_eyebrow_y, right_eyebrow_y)
        y_coords, _ = np.mgrid[0:H, 0:W]
        regions['forehead'] = skin_mask & (y_coords < eye_y_min - 5)

        # 3. Left Cheek
        left_cheek_poly = self._create_poly_mask((H, W), landmarks_cropped, self.mp_indices['left_cheek_poly'])
        regions['left_cheek'] = skin_mask & left_cheek_poly

        # 4. Right Cheek
        right_cheek_poly = self._create_poly_mask((H, W), landmarks_cropped, self.mp_indices['right_cheek_poly'])
        regions['right_cheek'] = skin_mask & right_cheek_poly


        for name, mask in regions.items():
            logger.info(f"Region '{name}' skin pixels: {np.sum(mask)}")
            
        return regions

    def _create_poly_mask(self, shape, landmarks, indices):
        mask = np.zeros(shape, dtype=np.uint8)
        pts = landmarks[indices].astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return mask > 0

    def _extract_mtcnn(self, parsing_mask, landmarks_cropped):
        logger.info("Extracting regions based on parsing mask and MTCNN landmarks...")
        H, W = parsing_mask.shape
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        skin_mask = (parsing_mask == self.class_skin)
        
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks_cropped[:5]
        
        eye_y_min = min(left_eye[1], right_eye[1])
        eye_y_max = max(left_eye[1], right_eye[1])
        mouth_y_max = max(left_mouth[1], right_mouth[1])
        
        nose_x = nose[0]

        regions = {}
        regions['nose'] = (parsing_mask == self.class_nose)
        regions['forehead'] = skin_mask & (y_coords < eye_y_min - 5)
        regions['left_cheek'] = skin_mask & (y_coords > eye_y_max + 5) & (y_coords < mouth_y_max + 15) & (x_coords < nose_x)
        regions['right_cheek'] = skin_mask & (y_coords > eye_y_max + 5) & (y_coords < mouth_y_max + 15) & (x_coords > nose_x)
        
        for name, mask in regions.items():
            logger.info(f"Region '{name}' skin pixels: {np.sum(mask)}")
            
        return regions
