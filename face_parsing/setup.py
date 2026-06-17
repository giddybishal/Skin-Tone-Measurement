import os
import gdown
import logging

logger = logging.getLogger(__name__)

def setup_bisenet_weights():
    # The Google Drive ID for the pretrained weights from zllrunning/face-parsing.PyTorch
    file_id = '154JgKpzCPW82qINcVieuPH3fZ2e0P812'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    output_dir = os.path.join(os.path.dirname(__file__), 'bisenet_repo', 'res', 'cp')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, '79999_iter.pth')
    
    if not os.path.exists(output_path):
        logger.info(f"Downloading BiSeNet weights to {output_path}...")
        gdown.download(url, output_path, quiet=False)
        logger.info("BiSeNet weights downloaded successfully.")
    else:
        logger.info("BiSeNet weights already exist.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_bisenet_weights()
