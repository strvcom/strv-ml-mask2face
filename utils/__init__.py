import numpy as np
from PIL import Image


def image_to_array(image: Image) -> np.ndarray:
    """Convert Image to array"""
    return np.asarray(image).astype('uint8')


def load_image(img_path: str) -> Image:
    """Load image to array"""
    return Image.open(img_path)
