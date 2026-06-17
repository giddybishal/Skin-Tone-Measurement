import sys
import os
import logging
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Add the cloned repo to sys.path
bisenet_repo_path = os.path.join(os.path.dirname(__file__), 'bisenet_repo')
if bisenet_repo_path not in sys.path:
    sys.path.append(bisenet_repo_path)

try:
    from model import BiSeNet
except ImportError:
    logger.error("Could not import BiSeNet. Ensure 'zllrunning/face-parsing.PyTorch' is cloned into 'bisenet_repo'.")

class FaceParser:
    def __init__(self, use_gpu: bool = False):
        """
        Initializes the BiSeNet Face Parser.
        """
        logger.info("Initializing FaceParser (BiSeNet)")
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        
        weight_path = os.path.join(bisenet_repo_path, 'res', 'cp', '79999_iter.pth')
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model weights not found at {weight_path}. Run setup.py first.")
        
        logger.info(f"Loading BiSeNet weights from {weight_path} to {self.device}")
        self.net.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        logger.info("FaceParser initialized successfully.")

    def parse_face(self, face_image):
        """
        Runs semantic segmentation on a cropped face image.
        
        Args:
            face_image: numpy array (RGB format, typical PIL image or cv2 converted)
            
        Returns:
            mask: numpy array (H, W) with integer class labels 0-18.
        """
        logger.info("Running face segmentation...")
        # BiSeNet expects 512x512 image for best results, but we can resize
        pil_img = Image.fromarray(face_image)
        original_size = pil_img.size # (W, H)
        
        img = pil_img.resize((512, 512), Image.BILINEAR)
        img = self.transform(img)
        img = torch.unsqueeze(img, 0)
        
        with torch.no_grad():
            img = img.to(self.device)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
        # Resize mask back to original face_image size
        parsing_pil = Image.fromarray(parsing.astype(np.uint8))
        parsing_pil = parsing_pil.resize(original_size, Image.NEAREST)
        
        return np.array(parsing_pil)
