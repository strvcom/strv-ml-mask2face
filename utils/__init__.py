import numpy as np
from PIL import Image
import requests
import functools
from tqdm.notebook import tqdm
import shutil

def image_to_array(image: Image) -> np.ndarray:
    """Convert Image to array"""
    return np.asarray(image).astype('uint8')


def load_image(img_path: str) -> Image:
    """Load image to array"""
    return Image.open(img_path)


def download_data(url, save_path, file_size=None):
    """Downloads data from `url` to `save_path`"""
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f'Request to {url} returned status code {r.status_code}')

    if file_size is None:
        file_size = int(r.headers.get('content-length', 0))

    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, 'read', total=file_size, desc='') as r_raw:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(r_raw, f)

def plot_image_triple():
    pass