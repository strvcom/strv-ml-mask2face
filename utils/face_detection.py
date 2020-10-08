"""Functions for face detection"""
from math import pi
from typing import Tuple, Optional, Dict

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
        # run detection and keep results with certain confidence only
        results = [item for item in detector.detect_faces(image_to_array(image)) if
                   item['confidence'] > minimal_confidence]

        # nothing found
        if len(results) == 0:
            return None

        # return result with highest confidence
        return max(results, key=lambda item: item['confidence'])

    # return function
    return get_keypoints


def plot_face_detection(image: Image, ax, face_keypoints_detecting_fun, hyp_ratio: float = 1 / 3):
    """Plot faces with keypoints and bounding boxes"""

    # detect keypoints
    keypoints = face_keypoints_detecting_fun(image)

    # make annotations
    if keypoints is not None:

        # get bounding box
        x, y, width, height = keypoints['box']

        # add rectangle patch for detected face
        rectangle = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rectangle)

        # add rectangle patch with slacks
        w_s, h_s = compute_slacks(height, width, hyp_ratio)
        rectangle = patches.Rectangle((x - w_s, y - h_s), width + 2 * w_s, height + 2 * h_s, linewidth=1, edgecolor='r',
                                      facecolor='none')
        ax.add_patch(rectangle)

        # add keypoints
        for coordinates in keypoints['keypoints'].values():
            circle = plt.Circle(coordinates, 3, color='r')
            ax.add_artist(circle)

    # add image
    ax.imshow(image)
