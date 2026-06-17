import logging
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, use_gpu: bool = False):
        """
        Initializes the face detector using MTCNN from facenet-pytorch.
        """
        logger.info("Initializing FaceDetector (MTCNN)")
        # keep_all=True ensures we get all faces so we can pick the largest
        device = 'cuda' if use_gpu else 'cpu'
        self.mtcnn = MTCNN(keep_all=True, device=device)
        logger.info("FaceDetector initialized successfully.")

    def detect_face(self, image):
        """
        Detects the primary face in an image.

        Args:
            image: numpy array (BGR format, typically loaded via cv2)
            
        Returns:
            dict containing 'bbox', 'landmarks' (as a numpy array (5,2)), and 'confidence', 
            or None if no face found.
        """
        logger.info("Detecting face...")
        
        # MTCNN works best with RGB PIL images
        # OpenCV loads in BGR, so we convert BGR -> RGB -> PIL
        rgb_image = image[:, :, ::-1] # BGR to RGB
        pil_img = Image.fromarray(rgb_image)
        
        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            logger.warning("No face detected in the image.")
            return None
        
        # Find the primary face (largest bounding box area)
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        max_idx = np.argmax(areas)
        
        bbox = boxes[max_idx].astype(int)
        confidence = probs[max_idx]
        face_landmarks = landmarks[max_idx].astype(int)
        
        logger.info(f"Face detected with confidence {confidence:.2f}")
        logger.info(f"Bounding box: x1={bbox[0]} y1={bbox[1]} x2={bbox[2]} y2={bbox[3]}")
        
        return {
            'bbox': bbox,
            'landmarks': face_landmarks,
            'confidence': confidence
        }
