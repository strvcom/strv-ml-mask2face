import copy
import dlib
import numpy as np
import os
import random
from PIL import Image, ImageFilter
from itertools import chain
from typing import Optional, Tuple

from utils import image_to_array
from utils.face_detection import compute_slacks, get_face_keypoints_detecting_function
from mask_utils.mask_utils import mask_image


def load_image(img_path: str) -> Image:
    """Load image to array"""
    return Image.open(img_path)


class MaskGeneratorArguments:
    # TODO: move to configuration
    def __init__(self):
        """Arguments for MaskTheFace mask generator"""
        self.mask_type = 'random'  # chose from ["surgical", "N95", "KN95", "cloth","random"]
        self.color = None  # string with hex color like #000000
        self.pattern = ''  # path to file with pattern
        self.pattern_weight = 0.9  # number from 0 to 1
        self.color_weight = 0.8  # number from 0 to 1
        self.filter_output = False  # Filter the image with mask on to make smother transitions


class DataGenerator:
    def __init__(self, configuration):
        self.configuration = configuration
        self.path_to_data = configuration.get('input_images_path')
        self.path_to_patterns = configuration.get('path_to_patterns')
        self.minimal_confidence = configuration.get('minimal_confidence')
        self.hyp_ratio = configuration.get('hyp_ratio')
        self.coordinates_range = configuration.get('coordinates_range')
        self.test_image_count = configuration.get('test_image_count')
        self.train_image_count = configuration.get('train_image_count')
        self.train_data_path = configuration.get('train_data_path')
        self.test_data_path = configuration.get('test_data_path')
        self.predictor = configuration.get('landmarks_predictor_path')
        # TODO: Check if predictor exists - if not download it here: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

        self.mask_gen_args = MaskGeneratorArguments()

        self.face_keypoints_detecting_fun = get_face_keypoints_detecting_function(self.minimal_confidence)

    @staticmethod
    def crop_face(image: Image, face_keypoints: Optional, hyp_ratio: float = 1 / 3) -> Image:
        """ Crop input image to just the face"""

        # no cropping - no face was detected
        if face_keypoints is None:
            return image

        # get bounding box
        x, y, width, height = face_keypoints['box']

        # compute slacks
        w_s, h_s = compute_slacks(height, width, hyp_ratio)

        # compute coordinates
        left = min(max(0, x - w_s), image.width)
        upper = min(max(0, y - h_s), image.height)
        right = min(x + width + w_s, image.width)
        lower = min(y + height + h_s, image.height)

        return image.crop((left, upper, right, lower))

    def get_face_landmarks(self, image):
        """Compute 68 facial landmarks"""
        landmarks = []
        image_array = image_to_array(image)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.predictor)
        face_rectangles = detector(image_array)
        if len(face_rectangles) < 1:
            return None
        dlib_shape = predictor(image_array, face_rectangles[0])
        for i in range(0, dlib_shape.num_parts):
            landmarks.append([dlib_shape.part(i).x, dlib_shape.part(i).y])
        return landmarks

    def get_files_faces(self):
        """Get path of all images in dataset"""
        return list(
            chain.from_iterable(
                [["{}/{}".format(folder, sub_folder) for sub_folder in
                  os.listdir(os.path.join(self.path_to_data, folder))]
                 for folder in os.listdir(self.path_to_data)]
            )
        )

    def generate_images(self, image_size=None, test_image_count=None, train_image_count=None):
        """Generate test and train data (images with and without the mask)"""
        if image_size is None:
            image_size = self.configuration.get('image_size')
        if test_image_count is None:
            test_image_count = self.test_image_count
        if train_image_count is None:
            train_image_count = self.train_image_count

        if not os.path.exists(self.train_data_path):
            os.mkdir(self.train_data_path)
            os.mkdir(os.path.join(self.train_data_path, 'inputs'))
            os.mkdir(os.path.join(self.train_data_path, 'outputs'))

        if not os.path.exists(self.test_data_path):
            os.mkdir(self.test_data_path)
            os.mkdir(os.path.join(self.test_data_path, 'inputs'))
            os.mkdir(os.path.join(self.test_data_path, 'outputs'))

        (test_in, test_out) = self.generate_data(test_image_count, image_size)
        (train_in, train_out) = self.generate_data(train_image_count, image_size)

        for i in range(len(test_in)):
            test_in[i].save(os.path.join(self.test_data_path, 'inputs', f"{i}.png"))
            test_out[i].save(os.path.join(self.test_data_path, 'outputs', f"{i}.png"))

        for i in range(len(train_in)):
            train_in[i].save(os.path.join(self.train_data_path, 'inputs', f"{i}.png"))
            train_out[i].save(os.path.join(self.train_data_path, 'outputs', f"{i}.png"))

    def generate_data(self, number_of_images, image_size=None):
        """Add masks on `number_of_images` images"""
        inputs = []
        outputs = []

        if image_size is None:
            image_size = self.configuration.get('image_size')

        for i, file in enumerate(random.sample(self.get_files_faces(), number_of_images)):
            # Load images
            image = load_image("{}/{}".format(self.path_to_data, file))

            # Detect keypoints and landmarks on face
            face_landmarks = self.get_face_landmarks(image)
            if face_landmarks is None:
                continue
            keypoints = self.face_keypoints_detecting_fun(image)

            # Genereate mask
            image_with_mask = mask_image(copy.deepcopy(image), face_landmarks, self.mask_gen_args)

            # Crop images
            cropped_image = self.crop_face(image_with_mask, keypoints)
            cropped_original = self.crop_face(image, keypoints)

            # Resize all images to NN input size
            res_image = cropped_image.resize(image_size)
            res_original = cropped_original.resize(image_size)

            # Save generated data to lists
            inputs.append(res_image)
            outputs.append(res_original)

            if i % 500 == 0:
                print(f'{i}/{number_of_images} images masked')

        return inputs, outputs
