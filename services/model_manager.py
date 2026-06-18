import os
import sys
import logging
import time

logger = logging.getLogger(__name__)

# Resolve project root relative to this file (services/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelManager:
    """
    Centralized model initialization and lifecycle management.

    All models are loaded once at startup and reused for every analysis request.
    BiSeNet weights are automatically downloaded if not present.
    """

    def __init__(self):
        self.detector = None
        self.parser = None
        self.extractor = None
        self.color_analyzer = None
        self.pipeline_config = {}
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def load_models(self):
        """
        Initialize all pipeline models. Call once at application startup.
        """
        t_start = time.perf_counter()
        logger.info("Loading all pipeline models...")

        # 1. Load pipeline config
        self._load_config()

        # 2. Ensure BiSeNet weights exist (auto-download if needed)
        self._ensure_bisenet_weights()

        # 3. Initialize face detector (MediaPipe — fast, dense landmarks)
        self._init_detector()

        # 4. Initialize face parser (BiSeNet)
        self._init_parser()

        # 5. Initialize region extractor
        self._init_extractor()

        # 6. Initialize color analyzer
        self._init_color_analyzer()

        self._loaded = True
        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(f"All models loaded successfully in {elapsed:.0f}ms")

    def _load_config(self):
        """Load pipeline configuration from YAML."""
        import yaml

        config_path = os.path.join(PROJECT_ROOT, 'config', 'pipeline.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.pipeline_config = yaml.safe_load(f) or {}
            logger.info(f"Pipeline config loaded from {config_path}")
        else:
            logger.warning(f"Pipeline config not found at {config_path}, using defaults")
            self.pipeline_config = {}

    def _ensure_bisenet_weights(self):
        """Check for BiSeNet weights and download automatically if missing."""
        weight_path = self._get_bisenet_weight_path()

        if os.path.exists(weight_path):
            logger.info(f"BiSeNet weights found at {weight_path}")
            return

        logger.info("BiSeNet weights not found. Downloading pretrained weights...")

        try:
            # Use the existing setup module
            sys.path.insert(0, PROJECT_ROOT)
            from face_parsing.setup import setup_bisenet_weights
            setup_bisenet_weights()

            if os.path.exists(weight_path):
                logger.info("BiSeNet weights downloaded successfully.")
            else:
                raise FileNotFoundError(
                    f"Weight download completed but file not found at {weight_path}"
                )
        except Exception as e:
            logger.error(f"Failed to download BiSeNet weights: {e}")
            raise RuntimeError(
                "Could not download BiSeNet pretrained weights. "
                "Please check your internet connection and try again. "
                f"Expected path: {weight_path}"
            ) from e

    def _get_bisenet_weight_path(self):
        """Get the expected path for BiSeNet weights."""
        return os.path.join(
            PROJECT_ROOT, 'face_parsing', 'bisenet_repo', 'res', 'cp', '79999_iter.pth'
        )

    def _init_detector(self):
        """Initialize MediaPipe face detector."""
        from face_detection.mediapipe_detector import MediaPipeDetector

        self.detector = MediaPipeDetector(use_gpu=False)
        logger.info("MediaPipe detector initialized.")

    def _init_parser(self):
        """Initialize BiSeNet face parser."""
        from face_parsing.bisenet_parser import FaceParser

        weight_path = self._get_bisenet_weight_path()
        self.parser = FaceParser(use_gpu=False, weight_path=weight_path)
        logger.info("BiSeNet parser initialized.")

    def _init_extractor(self):
        """Initialize region extractor."""
        from region_extraction.extractor import RegionExtractor

        self.extractor = RegionExtractor()
        logger.info("Region extractor initialized.")

    def _init_color_analyzer(self):
        """Initialize color analyzer with Monk mapping."""
        from skin_analysis.color_analyzer import ColorAnalyzer

        config_path = os.path.join(PROJECT_ROOT, 'config', 'monk_mapping.yaml')
        self.color_analyzer = ColorAnalyzer(config_path, self.pipeline_config)
        logger.info("Color analyzer initialized.")
