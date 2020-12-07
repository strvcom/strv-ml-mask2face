"""Functions to generate masks and place them on faces"""
import os
import random
import numpy as np
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFilter

# all allowed image types for patterns
ALLOWED_PATTERNS_TYPES = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')


def make_background_from_pattern(pattern_image: Image, width: int, height: int) -> Image:
    """Repeat the pattern so it can be used for mask"""

    result_image = Image.new('RGB', (width, height))
    x = 0
    while x < width:
        y = 0
        while y < height:
            result_image.paste(pattern_image, (x, y))
            y += pattern_image.size[1]
        x += pattern_image.size[0]
    return result_image


def load_mask_patterns(folder_with_patterns: str) -> List:
    """Load all pattern images"""
    return [Image.open("{}/{}".format(folder_with_patterns, file)) for file in os.listdir(folder_with_patterns)
            if file.lower().endswith(ALLOWED_PATTERNS_TYPES)]


def create_mask_keypoints_generator(coordinates_range: Tuple[int, int] = (-10, 10)):
    """"""

    def get_keypoints_for_mask(keypoints) -> Tuple:
        """Method to generate keypoints for mask"""

        # bounding box parameters
        x, y, width, height = keypoints['box']

        # coordinates of keypoints
        y_nose = keypoints['keypoints']['nose'][1]
        x_mouth_right = keypoints['keypoints']['mouth_right'][0]
        x_mouth_left = keypoints['keypoints']['mouth_left'][0]

        return (x + random.randint(*coordinates_range), y_nose + random.randint(*coordinates_range)), \
               (x + width + random.randint(*coordinates_range), y_nose + random.randint(*coordinates_range)), \
               (x_mouth_right + random.randint(*coordinates_range), y + height + random.randint(*coordinates_range)), \
               (x_mouth_left + random.randint(*coordinates_range), y + height + random.randint(*coordinates_range))

    # return the function
    return get_keypoints_for_mask


def get_face_with_mask(image: Image, pattern_images: List, face_keypoints: Optional, keypoints_for_mask_fun) -> Image:
    """Add mask to a face on the image"""

    # do not add mask to the image
    if face_keypoints is None:
        return image

    # generate mask coordinates
    mask_keypoints = keypoints_for_mask_fun(face_keypoints)

    # prepare a pattern for the mask
    mask_pattern = make_background_from_pattern(pattern_images[random.randint(0, len(pattern_images) - 1)], *image.size)

    # create the mask on face
    mask = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(mask)
    draw.polygon(mask_keypoints, fill=(255, 255, 0))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    
    # create the mask of the mask (1 - mask, 0 - no mask)
    mask_mask = Image.new("L", image.size, (0))
    draw = ImageDraw.Draw(mask_mask)
    draw.polygon(mask_keypoints, fill=(1))
    mask_values = np.array(mask_mask)

    # final image is composite of mask and original image
    return (Image.composite(mask_pattern, image, mask), mask_values)
