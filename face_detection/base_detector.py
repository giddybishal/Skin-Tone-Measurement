from abc import ABC, abstractmethod
import logging

class FaceDetectorBase(ABC):
    """
    Abstract base class for face detectors.
    """
    
    @abstractmethod
    def detect_face(self, image):
        """
        Detects the primary face in an image.

        Args:
            image: numpy array (BGR format)
            
        Returns:
            dict containing:
              - 'bbox': [x1, y1, x2, y2]
              - 'landmarks': numpy array of landmarks
              - 'confidence': float (optional)
            or None if no face found.
        """
        pass
