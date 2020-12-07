import os
import numpy as np
import cv2
from datetime import datetime
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.utils import CustomObjectScope


class Mask2FaceModel():
    """ Model for Mask2Face - removes mask from people faces using U-net neural network
    """
    
    def __init__(self, input_size=(256, 224)):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.input_size = input_size
        self.model = self.build_model()
    
    def load_model(self, model_path):
        with CustomObjectScope({'iou': Mask2FaceModel.iou}):
            self.model = tf.keras.models.load_model(model_path)
        
    def build_model(self):
        num_filters = [16, 32, 48, 64]
        inputs = Input((self.input_size[0], self.input_size[1], 3))

        skip_x = []
        x = inputs
        
        ## Encoder
        for f in num_filters:
            x = Mask2FaceModel.conv_block(x, f)
            skip_x.append(x)
            x = MaxPool2D((2, 2))(x)

        ## Bridge
        x = Mask2FaceModel.conv_block(x, num_filters[-1])

        num_filters.reverse()
        skip_x.reverse()
        ## Decoder
        for i, f in enumerate(num_filters):
            x = UpSampling2D((2, 2))(x)
            xs = skip_x[i]
            x = Concatenate()([x, xs])
            x = Mask2FaceModel.conv_block(x, f)

        ## Output
        x = Conv2D(1, (1, 1), padding="same")(x)
        x = Activation("sigmoid")(x)

        return Model(inputs, x)
    
    def train(self, epochs=20, batch_size=20, learning_rate=1e-4, loss_function='MSE'):
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = Mask2FaceModel.load_train_data()

        train_dataset = Mask2FaceModel.tf_dataset(train_x, train_y, batch=batch_size)
        valid_dataset = Mask2FaceModel.tf_dataset(valid_x, valid_y, batch=batch_size)

        opt = tf.keras.optimizers.Adam(learning_rate)
        metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), Mask2FaceModel.iou]
        self.model.compile(loss=loss_function, optimizer=opt, metrics=metrics)

        callbacks = [
            ModelCheckpoint(f'model_epochs-{epochs}_batch-{batch_size}_loss-{loss_function}_{Mask2FaceModel.get_datetime_string()}.h5'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
            CSVLogger("data.csv"),
            TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
        ]

        train_steps = len(train_x)//batch_size
        valid_steps = len(valid_x)//batch_size

        if len(train_x) % batch_size != 0:
            train_steps += 1
        if len(valid_x) % batch_size != 0:
            valid_steps += 1

        self.model.fit(train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks)
    
    def predict_batch(self, batch_size=8):
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = Mask2FaceModel.load_test_data()
        test_dataset = Mask2FaceModel.tf_dataset(test_x, test_y, batch=batch_size)

        test_steps = (len(test_x)//batch_size)
        if len(test_x) % batch_size != 0:
            test_steps += 1
        
        result_dir = f"data/results/{Mask2FaceModel.get_datetime_string()}/"
        os.mkdir(result_dir)

        for i, (x, y) in enumerate(zip(test_x, test_y)):
            x = Mask2FaceModel.read_image(x)
            y = Mask2FaceModel.read_mask(y, False)
            y_pred = self.model.predict(np.expand_dims(x, axis=0))
            h, w, _ = x.shape
            white_line = np.ones((h, 10, 3)) * 255.0

            all_images = [
                x * 255.0, white_line,
                Mask2FaceModel.mask_parse(y), white_line,
                Mask2FaceModel.mask_parse(y_pred) * 255.0
            ]
            image = np.concatenate(all_images, axis=1)
            cv2.imwrite(os.path.join(result_dir, f"{i}.png"), image)
        
    def summary(self):
        self.model.summary()
    
    @staticmethod
    def conv_block(x, num_filters):
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
        
    @staticmethod
    def get_datetime_string():
        now = datetime.now()
        return now.strftime("%Y%m%d_%H_%M_%S")

    @staticmethod
    def load_train_data(split=0.2):
        return Mask2FaceModel.load_data("data/train/inputs", "data/train/outputs", split)

    @staticmethod
    def load_test_data(split=0.2):
        return Mask2FaceModel.load_data("data/test/inputs", "data/test/outputs", split)

    @staticmethod
    def load_data(input_path, output_path, split=0.2):
        images = sorted(glob(os.path.join(input_path, "*.png")))
        masks = sorted(glob(os.path.join(output_path, "*.png")))
        
        total_size = len(images)
        valid_size = int(split * total_size)
        test_size = int(split * total_size)

        train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
        train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

        train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
        train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

    @staticmethod
    def read_image(path):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (256, 256))
        x = x/255.0
        return x

    @staticmethod
    def read_mask(path, normalize=True):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (256, 256))
        if normalize:
            x = x/255.0
        x = np.expand_dims(x, axis=-1)
        return x

    @staticmethod
    def tf_parse(x, y):
        def _parse(x, y):
            x = Mask2FaceModel.read_image(x.decode())
            y = Mask2FaceModel.read_mask(y.decode())
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
        x.set_shape([256, 256, 3])
        y.set_shape([256, 256, 1])
        return x, y

    @staticmethod
    def tf_dataset(x, y, batch=8):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(Mask2FaceModel.tf_parse)
        dataset = dataset.batch(batch)
        dataset = dataset.repeat()
        return dataset

    @staticmethod
    def mask_parse(mask):
        mask = np.squeeze(mask)
        mask = [mask, mask, mask]
        mask = np.transpose(mask, (1, 2, 0))
        return mask

    @staticmethod
    def iou(y_true, y_pred):
        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + 1e-15) / (union + 1e-15)
            x = x.astype(np.float32)
            return x
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)
    

if __name__ == "__main__":
    model = Mask2FaceModel()
    model.summary()