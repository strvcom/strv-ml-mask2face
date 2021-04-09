import os
from datetime import datetime
from glob import glob
from typing import Tuple, Optional
from utils import load_image
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import CustomObjectScope
from utils.face_detection import get_face_keypoints_detecting_function, crop_face, get_crop_points
from utils.architectures import UNet
from tensorflow.keras.losses import MeanSquaredError, mean_squared_error


class Mask2FaceModel(tf.keras.models.Model):
    """
    Model for Mask2Face - removes mask from people faces using U-net neural network
    """
    def __init__(self, model: tf.keras.models.Model, configuration=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: tf.keras.models.Model = model
        self.configuration = configuration
        self.face_keypoints_detecting_fun = get_face_keypoints_detecting_function(0.8)
        self.mse = MeanSquaredError()

    def call(self, x, **kwargs):
        return self.model(x)

    @staticmethod
    @tf.function
    def ssim_loss(gt, y_pred, max_val=1.0):
        """
        Computes standard SSIM loss
        @param gt: Ground truth image
        @param y_pred: Predicted image
        @param max_val: Maximal SSIM value
        @return: SSIM loss
        """
        return 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))

    @staticmethod
    @tf.function
    def ssim_l1_loss(gt, y_pred, max_val=1.0, l1_weight=1.0):
        """
        Computes SSIM loss with L1 normalization
        @param gt: Ground truth image
        @param y_pred: Predicted image
        @param max_val: Maximal SSIM value
        @param l1_weight: Weight of L1 normalization
        @return: SSIM L1 loss
        """
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))
        l1 = mean_squared_error(gt, y_pred)
        return ssim_loss + tf.cast(l1 * l1_weight, tf.float32)

    @staticmethod
    def load_model(model_path, configuration=None):
        """
        Loads saved h5 file with trained model.
        @param configuration: Optional instance of Configuration with config JSON
        @param model_path: Path to h5 file
        @return: Mask2FaceModel
        """
        with CustomObjectScope({'ssim_loss': Mask2FaceModel.ssim_loss, 'ssim_l1_loss': Mask2FaceModel.ssim_l1_loss}):
            model = tf.keras.models.load_model(model_path)
        return Mask2FaceModel(model, configuration)

    @staticmethod
    def build_model(architecture: UNet, input_size: Tuple[int, int, int], filters: Optional[Tuple] = None,
                    kernels: Optional[Tuple] = None, configuration=None):
        """
        Builds model based on input arguments
        @param architecture: utils.architectures.UNet architecture
        @param input_size: Size of input images
        @param filters: Tuple with sizes of filters in U-net
        @param kernels: Tuple with sizes of kernels in U-net. Must be the same size as filters.
        @param configuration: Optional instance of Configuration with config JSON
        @return: Mask2FaceModel
        """
        return Mask2FaceModel(architecture.build_model(input_size, filters, kernels).get_model(), configuration)

    def train(self, epochs=20, batch_size=20, loss_function='mse', learning_rate=1e-4,
              predict_difference: bool = False):
        """
        Train the model.
        @param epochs: Number of epochs during training
        @param batch_size: Batch size
        @param loss_function: Loss function. Either standard tensorflow loss function or `ssim_loss` or `ssim_l1_loss`
        @param learning_rate: Learning rate
        @param predict_difference: Compute prediction on difference between input and output image
        @return: History of training
        """
        # get data
        (train_x, train_y), (valid_x, valid_y) = self.load_train_data()
        (test_x, test_y) = self.load_test_data()

        train_dataset = Mask2FaceModel.tf_dataset(train_x, train_y, batch_size, predict_difference)
        valid_dataset = Mask2FaceModel.tf_dataset(valid_x, valid_y, batch_size, predict_difference, train=False)
        test_dataset = Mask2FaceModel.tf_dataset(test_x, test_y, batch_size, predict_difference, train=False)

        # select loss
        if loss_function == 'ssim_loss':
            loss = Mask2FaceModel.ssim_loss
        elif loss_function == 'ssim_l1_loss':
            loss = Mask2FaceModel.ssim_l1_loss
        else:
            loss = loss_function

        # compile loss with selected loss function
        self.model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        )

        # define callbacks
        callbacks = [
            ModelCheckpoint(
                f'models/model_epochs-{epochs}_batch-{batch_size}_loss-{loss_function}_{Mask2FaceModel.get_datetime_string()}.h5'),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        # evaluation before training
        results = self.model.evaluate(test_dataset)
        print("- TEST -> LOSS: {:10.4f}, ACC: {:10.4f}, RECALL: {:10.4f}, PRECISION: {:10.4f}".format(*results))

        # fit the model
        history = self.model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=callbacks)

        # evaluation after training
        results = self.model.evaluate(test_dataset)
        print("- TEST -> LOSS: {:10.4f}, ACC: {:10.4f}, RECALL: {:10.4f}, PRECISION: {:10.4f}".format(*results))

        # use the model for inference on several test images
        self._test_results(test_x, test_y, predict_difference)

        # return training history
        return history

    def _test_results(self, test_x, test_y, predict_difference: bool):
        """
        Test trained model on testing dataset. All images in testing dataset are processed and result image triples
        (input with mask, ground truth, model output) are stored to `data/results` into folder with time stamp
        when this method was executed.
        @param test_x: List of input images
        @param test_y: List of ground truth output images
        @param predict_difference: Compute prediction on difference between input and output image
        @return: None
        """
        if self.configuration is None:
            result_dir = f'data/results/{Mask2FaceModel.get_datetime_string()}/'
        else:
            result_dir = os.path.join(self.configuration.get('test_results_dir'), Mask2FaceModel.get_datetime_string())
        os.makedirs(result_dir, exist_ok=True)

        for i, (x, y) in enumerate(zip(test_x, test_y)):
            x = Mask2FaceModel.read_image(x)
            y = Mask2FaceModel.read_image(y)

            y_pred = self.model.predict(np.expand_dims(x, axis=0))
            if predict_difference:
                y_pred = (y_pred * 2) - 1
                y_pred = np.clip(x - y_pred.squeeze(axis=0), 0.0, 1.0)
            else:
                y_pred = y_pred.squeeze(axis=0)
            h, w, _ = x.shape
            white_line = np.ones((h, 10, 3)) * 255.0

            all_images = [
                x * 255.0, white_line,
                y * 255.0, white_line,
                y_pred * 255.0
            ]
            image = np.concatenate(all_images, axis=1)
            cv2.imwrite(os.path.join(result_dir, f"{i}.png"), image)

    def summary(self):
        """
        Prints model summary
        """
        self.model.summary()

    def predict(self, img_path, predict_difference: bool = False):
        """
        Use trained model to take down the mask from image with person wearing the mask.
        @param img_path: Path to image to processed
        @param predict_difference: Compute prediction on difference between input and output image
        @return: Image without the mask on the face
        """
        # Load image into RGB format
        image = load_image(img_path)
        image = image.convert('RGB')

        # Find facial keypoints and crop the image to just the face
        keypoints = self.face_keypoints_detecting_fun(image)
        cropped_image = crop_face(image, keypoints)
        print(cropped_image.size)

        # Resize image to input recognized by neural net
        resized_image = cropped_image.resize((256, 256))
        image_array = np.array(resized_image)

        # Convert from RGB to BGR (open cv format)
        image_array = image_array[:, :, ::-1].copy()
        image_array = image_array / 255.0

        # Remove mask from input image
        y_pred = self.model.predict(np.expand_dims(image_array, axis=0))
        h, w, _ = image_array.shape

        if predict_difference:
            y_pred = (y_pred * 2) - 1
            y_pred = np.clip(image_array - y_pred.squeeze(axis=0), 0.0, 1.0)
        else:
            y_pred = y_pred.squeeze(axis=0)

        # Convert output from model to image and scale it back to original size
        y_pred = y_pred * 255.0
        im = Image.fromarray(y_pred.astype(np.uint8)[:, :, ::-1])
        im = im.resize(cropped_image.size)
        left, upper, _, _ = get_crop_points(image, keypoints)

        # Combine original image with output from model
        image.paste(im, (int(left), int(upper)))
        return image

    @staticmethod
    def get_datetime_string():
        """
        Creates date-time string
        @return: String with current date and time
        """
        now = datetime.now()
        return now.strftime("%Y%m%d_%H_%M_%S")

    def load_train_data(self, split=0.2):
        """
        Loads training data (paths to training images)
        @param split: Percentage of training data used for validation as float from 0.0 to 1.0. Default 0.2.
        @return: Two tuples - first with training data (tuple with (input images, output images)) and second
                    with validation data (tuple with (input images, output images))
        """
        if self.configuration is None:
            train_dir = 'data/train/'
            limit = None
        else:
            train_dir = self.configuration.get('train_data_path')
            limit = self.configuration.get('train_data_limit')
        print(f'Loading training data from {train_dir} with limit of {limit} images')
        return Mask2FaceModel.load_data(os.path.join(train_dir, 'inputs'), os.path.join(train_dir, 'outputs'), split, limit)

    def load_test_data(self):
        """
        Loads testing data (paths to testing images)
        @return: Tuple with testing data - (input images, output images)
        """
        if self.configuration is None:
            test_dir = 'data/test/'
            limit = None
        else:
            test_dir = self.configuration.get('test_data_path')
            limit = self.configuration.get('test_data_limit')
        print(f'Loading testing data from {test_dir} with limit of {limit} images')
        return Mask2FaceModel.load_data(os.path.join(test_dir, 'inputs'), os.path.join(test_dir, 'outputs'), None, limit)

    @staticmethod
    def load_data(input_path, output_path, split=0.2, limit=None):
        """
        Loads data (paths to images)
        @param input_path: Path to folder with input images
        @param output_path: Path to folder with output images
        @param split: Percentage of data used for validation as float from 0.0 to 1.0. Default 0.2.
                      If split is None it expects you are loading testing data, otherwise expects training data.
        @param limit: Maximal number of images loaded from data folder. Default None (no limit).
        @return: If split is not None: Two tuples - first with training data (tuple with (input images, output images))
                    and second with validation data (tuple with (input images, output images))
                 Else: Tuple with testing data - (input images, output images)
        """
        images = sorted(glob(os.path.join(input_path, "*.png")))
        masks = sorted(glob(os.path.join(output_path, "*.png")))
        if len(images) == 0:
            raise TypeError(f'No images found in {input_path}')
        if len(masks) == 0:
            raise TypeError(f'No images found in {output_path}')

        if limit is not None:
            images = images[:limit]
            masks = masks[:limit]

        if split is not None:
            total_size = len(images)
            valid_size = int(split * total_size)
            train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
            train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)
            return (train_x, train_y), (valid_x, valid_y)

        else:
            return images, masks

    @staticmethod
    def read_image(path):
        """
        Loads image, resize it to size 256x256 and normalize to float values from 0.0 to 1.0.
        @param path: Path to image to be loaded.
        @return: Loaded image in open CV format.
        """
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (256, 256))
        x = x / 255.0
        return x

    @staticmethod
    def tf_parse(x, y):
        """
        Mapping function for dataset creation. Load and resize images.
        @param x: Path to input image
        @param y: Path to output image
        @return: Tuple with input and output image with shape (256, 256, 3)
        """
        def _parse(x, y):
            x = Mask2FaceModel.read_image(x.decode())
            y = Mask2FaceModel.read_image(y.decode())
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
        x.set_shape([256, 256, 3])
        y.set_shape([256, 256, 3])
        return x, y

    @staticmethod
    def tf_dataset(x, y, batch=8, predict_difference: bool = False, train: bool = True):
        """
        Creates standard tensorflow dataset.
        @param x: List of paths to input images
        @param y: List of paths to output images
        @param batch: Batch size
        @param predict_difference: Compute prediction on difference between input and output image
        @param train: Flag if training dataset should be generated
        @return: Dataset with loaded images
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(Mask2FaceModel.tf_parse)
        random_seed = random.randint(0, 999999999)

        if predict_difference:
            def map_output(img_in, img_target):
                return img_in, (img_in - img_target + 1.0) / 2.0

            dataset = dataset.map(map_output)

        if train:
            # for the train set, we want to apply data augmentations and shuffle data to different batches

            # random flip
            def flip(img_in, img_out):
                return tf.image.random_flip_left_right(img_in, random_seed), \
                       tf.image.random_flip_left_right(img_out, random_seed)

            # augmenting quality - parameters
            hue_delta = 0.05
            saturation_low = 0.2
            saturation_up = 1.3
            brightness_delta = 0.1
            contrast_low = 0.2
            contrast_up = 1.5

            # augmenting quality
            def color(img_in, img_out):
                img_in = tf.image.random_hue(img_in, hue_delta, random_seed)
                img_in = tf.image.random_saturation(img_in, saturation_low, saturation_up, random_seed)
                img_in = tf.image.random_brightness(img_in, brightness_delta, random_seed)
                img_in = tf.image.random_contrast(img_in, contrast_low, contrast_up, random_seed)
                img_out = tf.image.random_hue(img_out, hue_delta, random_seed)
                img_out = tf.image.random_saturation(img_out, saturation_low, saturation_up, random_seed)
                img_out = tf.image.random_brightness(img_out, brightness_delta, random_seed)
                img_out = tf.image.random_contrast(img_out, contrast_low, contrast_up, random_seed)
                return img_in, img_out

            # shuffle data and create batches
            dataset = dataset.shuffle(5000)
            dataset = dataset.batch(batch)

            # apply augmentations
            dataset = dataset.map(flip)
            dataset = dataset.map(color)
        else:
            dataset = dataset.batch(batch)

        return dataset.prefetch(tf.data.experimental.AUTOTUNE)
