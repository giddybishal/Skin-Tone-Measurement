import logging
import cv2
import numpy as np
import mediapipe as mp
from .base_detector import FaceDetectorBase

logger = logging.getLogger(__name__)

class MediaPipeDetector(FaceDetectorBase):
    def __init__(self, use_gpu: bool = False):
        """
        Initializes the face detector using MediaPipe Face Mesh.
        Note: MediaPipe CPU/GPU backend is typically handled automatically,
        but we ignore use_gpu for standard python usage.
        """
        logger.info("Initializing FaceDetector (MediaPipe)")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
        logger.info("MediaPipe FaceDetector initialized successfully.")

    def detect_face(self, image):
        """
        Detects the primary face in an image using MediaPipe.

        Args:
            image: numpy array (BGR format)
            
        Returns:
            dict containing 'bbox', 'landmarks' (as a numpy array (468, 2)), and 'confidence',
            or None if no face found.
        """
        logger.info("Detecting face with MediaPipe...")
        
        # MediaPipe expects RGB images
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            logger.warning("No face detected by MediaPipe.")
            return None
        
        # We only process the first face (max_num_faces=1)
        face_landmarks = results.multi_face_landmarks[0]
        
        h, w = image.shape[:2]
        
        landmarks_points = []
        x_coords = []
        y_coords = []
        
        for landmark in face_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_points.append([x, y])
            x_coords.append(x)
            y_coords.append(y)
            
        landmarks_np = np.array(landmarks_points)
        
        # Bounding box from landmarks
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        
        # Add slight padding to the bbox since landmarks don't cover the full head
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((y_max - y_min) * 0.1)
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y * 2) # More padding on top for forehead
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)
        
        bbox = np.array([x_min, y_min, x_max, y_max])
        
        logger.info(f"Face detected (MediaPipe)")
        logger.info(f"Bounding box: x1={bbox[0]} y1={bbox[1]} x2={bbox[2]} y2={bbox[3]}")
        
        return {
            'bbox': bbox,
            'landmarks': landmarks_np,
            'confidence': 1.0 # MediaPipe doesn't return confidence per face
        }
