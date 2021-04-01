import copy
import dlib
import numpy as np
import os
import bz2
import random
from PIL import Image, ImageFilter
from itertools import chain
from typing import Optional, Tuple
import glob
import functools
import shutil
import requests
from tqdm.notebook import tqdm
from utils import image_to_array, load_image
from utils.face_detection import crop_face, get_face_keypoints_detecting_function
from mask_utils.mask_utils import mask_image


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
        self.predictor_path = configuration.get('landmarks_predictor_path')
        self.check_predictor()

        self.valid_image_extensions = ('png', 'jpg', 'jpeg')
        self.face_keypoints_detecting_fun = get_face_keypoints_detecting_function(self.minimal_confidence)

    def check_predictor(self):
        """ Check if predictor exists. If not downloads it. """
        if not os.path.exists(self.predictor_path):
            url = self.configuration.get('landmarks_predictor_download_url')
            r = requests.get(url, stream=True, allow_redirects=True)
            if r.status_code != 200:
                r.raise_for_status()
                raise RuntimeError(f'Request to {url} returned status code {r.status_code}')
            r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
            with tqdm.wrapattr(r.raw, "read", total=64040097, desc="") as r_raw:
                with open(self.predictor_path + '.bz2', "wb") as f:
                    shutil.copyfileobj(r_raw, f)
            print(f'Decompressing downloaded file into {self.predictor_path}')
            with bz2.BZ2File(self.predictor_path + '.bz2') as fr, open(self.predictor_path, 'wb') as fw:
                shutil.copyfileobj(fr, fw)

    def get_face_landmarks(self, image):
        """Compute 68 facial landmarks"""
        landmarks = []
        image_array = image_to_array(image)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.predictor_path)
        face_rectangles = detector(image_array)
        if len(face_rectangles) < 1:
            return None
        dlib_shape = predictor(image_array, face_rectangles[0])
        for i in range(0, dlib_shape.num_parts):
            landmarks.append([dlib_shape.part(i).x, dlib_shape.part(i).y])
        return landmarks

    def get_files_faces(self):
        """Get path of all images in dataset"""
        files = []
        for dirpath, dirs, files in os.walk(self.path_to_data):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                if fname.endswith(self.valid_image_extensions):
                    files.append(fname)

        return files

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

        self.generate_data(test_image_count,
                           image_size=image_size,
                           save_to=self.test_data_path)
        self.generate_data(train_image_count,
                           image_size=image_size,
                           save_to=self.train_data_path)

    def generate_data(self, number_of_images, image_size=None, save_to=None):
        """ Add masks on `number_of_images` images
            if save_to is valid path to folder images are saved there otherwise generated data are just returned in list
        """
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

            # Generate mask
            image_with_mask = mask_image(copy.deepcopy(image), face_landmarks, self.configuration)

            # Crop images
            cropped_image = crop_face(image_with_mask, keypoints)
            cropped_original = crop_face(image, keypoints)

            # Resize all images to NN input size
            res_image = cropped_image.resize(image_size)
            res_original = cropped_original.resize(image_size)

            # Save generated data to lists or to folder
            if save_to is None:
                inputs.append(res_image)
                outputs.append(res_original)
            else:
                res_image.save(os.path.join(save_to, 'inputs', f"{i:06d}.png"))
                res_original.save(os.path.join(save_to, 'outputs', f"{i:06d}.png"))

            if i % 500 == 0:
                print(f'{i}/{number_of_images} images masked')

        if save_to is None:
            return inputs, outputs
