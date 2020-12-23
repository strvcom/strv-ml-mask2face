import os
import numpy as np
import random
from itertools import chain
from utils import load_image
from PIL import Image
from typing import Optional, Tuple

from utils.face_detection import compute_slacks, get_face_keypoints_detecting_function
from utils.mask_generator import load_mask_patterns, create_mask_keypoints_generator, get_face_with_mask


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

        self.patterns = load_mask_patterns(self.path_to_patterns)
        self.face_keypoints_detecting_fun = get_face_keypoints_detecting_function(self.minimal_confidence)
        self.keypoints_for_mask_fun = create_mask_keypoints_generator(self.coordinates_range)

    def crop_face(self, image: Image, face_keypoints: Optional, hyp_ratio: float = 1/3) -> Image:

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

    def crop_mask(self, mask, face_keypoints: Optional, hyp_ratio: float = 1/3):
        if face_keypoints is None:
            return mask
        x, y, width, height = face_keypoints['box']
        w_s, h_s = compute_slacks(height, width, hyp_ratio)
        return mask[max(0, y - int(h_s)):y + height + int(h_s), max(0, x - int(w_s)):x + width + int(w_s)]

    def pad_and_resize_image(self, image: Image, height: int = 256, width: int = 256):

        # compute ratio of current height and width to target
        h_ratio = image.height / height
        w_ratio = image.width / width

        # resize image if any side is greater then target
        if h_ratio > 1 or w_ratio > 1:
            if h_ratio > w_ratio:
                print()
            else:
                print()

    def get_files_faces(self):
        return list(
            chain.from_iterable(
                [["{}/{}".format(folder, sub_folder) for sub_folder in os.listdir(os.path.join(self.path_to_data, folder))]
                 for folder in os.listdir(self.path_to_data)]
            )
        )

    def generate_images(self, image_size=None, test_image_count=None, train_image_count=None):
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
        inputs = []
        outputs = []

        if image_size is None:
            image_size = self.configuration.get('image_size')

        for i, file in enumerate(random.sample(self.get_files_faces(), number_of_images)):
            # Load images
            image = load_image("{}/{}".format(self.path_to_data, file))

            # Detect keypoints on face
            keypoints = self.face_keypoints_detecting_fun(image)

            # Genereate mask
            (image_with_mask, mask_values) = get_face_with_mask(image, self.patterns, keypoints, self.keypoints_for_mask_fun)

            # Crop images
            cropped_image = self.crop_face(image_with_mask, keypoints)
            cropped_original = self.crop_face(image, keypoints)
            cropped_mask = self.crop_mask(mask_values, keypoints)

            # Resize all images to NN input size
            res_image = cropped_image.resize(image_size)
            res_original = cropped_original.resize(image_size)
            mask_image = Image.fromarray(cropped_mask)
            mask_image = mask_image.resize(image_size)
            res_mask = np.array(mask_image)

            # Save generated data to lists
            inputs.append(res_image)
            outputs.append(res_original)

        return inputs, outputs
