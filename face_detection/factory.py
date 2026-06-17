import logging
from .mtcnn_detector import MTCNNDetector
from .mediapipe_detector import MediaPipeDetector

logger = logging.getLogger(__name__)

def get_detector(backend_name: str, use_gpu: bool = False):
    """
    Returns the appropriate face detector based on the backend name.
    
    Args:
        backend_name (str): 'mtcnn' or 'mediapipe'
        use_gpu (bool): whether to use GPU (applicable to MTCNN)
        
    Returns:
        FaceDetectorBase instance
    """
    backend_name = backend_name.lower().strip()
    
    if backend_name == 'mtcnn':
        return MTCNNDetector(use_gpu=use_gpu)
    elif backend_name == 'mediapipe':
        return MediaPipeDetector(use_gpu=use_gpu)
    else:
        logger.error(f"Unknown detector backend: {backend_name}. Falling back to mtcnn.")
        return MTCNNDetector(use_gpu=use_gpu)
