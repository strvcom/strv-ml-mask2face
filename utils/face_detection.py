"""Functions for face detection"""
from math import pi
from typing import Tuple, Optional, Dict

import tensorflow as tf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from mtcnn import MTCNN
from trianglesolver import solve

from utils import image_to_array


def compute_slacks(height, width, hyp_ratio) -> Tuple[float, float]:
    """Compute slacks to add to bounding box on each site"""

    # compute angle and side for hypotenuse
    _, b, _, A, _, _ = solve(c=width, a=height, B=pi / 2)

    # compute new height and width
    a, _, c, _, _, _ = solve(b=b * (1.0 + hyp_ratio), B=pi / 2, A=A)

    # compute slacks
    return c - width, a - height


def get_face_keypoints_detecting_function(minimal_confidence: float = 0.8):
    """Create function for face keypoints detection"""

    # face detector
    detector = MTCNN()

    # detect faces and their keypoints
    def get_keypoints(image: Image) -> Optional[Dict]:

        # run inference to detect faces (on CPU only)
        with tf.device("/cpu:0"):
            detection = detector.detect_faces(image_to_array(image))

        # run detection and keep results with certain confidence only
        results = [item for item in detection if item['confidence'] > minimal_confidence]

        # nothing found
        if len(results) == 0:
            return None

        # return result with highest confidence and size
        return max(results, key=lambda item: item['confidence'] * item['box'][2] * item['box'][3])

    # return function
    return get_keypoints


def plot_face_detection(image: Image, ax, face_keypoints: Optional, hyp_ratio: float = 1 / 3):
    """Plot faces with keypoints and bounding boxes"""

    # make annotations
    if face_keypoints is not None:

        # get bounding box
        x, y, width, height = face_keypoints['box']

        # add rectangle patch for detected face
        rectangle = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rectangle)

        # add rectangle patch with slacks
        w_s, h_s = compute_slacks(height, width, hyp_ratio)
        rectangle = patches.Rectangle((x - w_s, y - h_s), width + 2 * w_s, height + 2 * h_s, linewidth=1, edgecolor='r',
                                      facecolor='none')
        ax.add_patch(rectangle)

        # add keypoints
        for coordinates in face_keypoints['keypoints'].values():
            circle = plt.Circle(coordinates, 3, color='r')
            ax.add_artist(circle)

    # add image
    ax.imshow(image)


def get_crop_points(image: Image, face_keypoints: Optional, hyp_ratio: float = 1 / 3) -> Image:
    """Find position where to crop face from image"""
    if face_keypoints is None:
        return 0, 0, image.width, image.height

    # get bounding box
    x, y, width, height = face_keypoints['box']

    # compute slacks
    w_s, h_s = compute_slacks(height, width, hyp_ratio)

    # compute coordinates
    left = min(max(0, x - w_s), image.width)
    upper = min(max(0, y - h_s), image.height)
    right = min(x + width + w_s, image.width)
    lower = min(y + height + h_s, image.height)

    return left, upper, right, lower


def crop_face(image: Image, face_keypoints: Optional, hyp_ratio: float = 1 / 3) -> Image:
    """Crop input image to just the face"""
    if face_keypoints is None:
        print("No keypoints detected on image")
        return image

    left, upper, right, lower = get_crop_points(image, face_keypoints, hyp_ratio)

    return image.crop((left, upper, right, lower))
