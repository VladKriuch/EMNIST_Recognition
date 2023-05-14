# Built in libraries
import os
import argparse
import csv
from io import StringIO

# Machine Learning libraries
import tensorflow as tf

# Image preprocessing
import PIL.Image

# Math
import numpy as np

# local
from mapping import MAPPER


class ImageProcessor:
    """
    Class for processing one image
    Might be done via DataProcessGenerator but I dont know what model will I use yet and how the inputs will look like
    """
    def __init__(self, target_size):
        self.target_size = target_size

    def process_image(self, path_to_image: str = ""):
        # Load image with tf.keras
        image = tf.keras.preprocessing.image.load_img(path_to_image)
        image = tf.keras.preprocessing.image.img_to_array(image)

        # If image in rgb channel - use rgb_to_grayscale
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)

        # Convert image to just zeros and ones as it were in dataset
        image = tf.where(image > 127, 255 * tf.ones_like(image), tf.zeros_like(image))\
            .numpy().reshape(image.shape[0], image.shape[1])

        # Resize image via PILLOW
        image = PIL.Image.fromarray(image.astype(np.uint8), mode='L')
        image = image.resize(self.target_size)
        image = np.array(image).astype(np.float16)

        # Normalize to 0 / 1
        image /= 255
        # print(image.shape)

        return image


class Program:
    """
    Class for program solution. Combines everything that is needed.
    P.S. One class for everything is bad approach, but for test task I believe it is ok to use it.
    """

    def __init__(self, path_to_weights_file: str = "", path_to_images: str = ""):
        """

        :param path_to_weights_file: path to keras file with saved model weights
        :param path_to_images: path to folder with images
        """
        if self.validate_path_to_images(path_to_images):
            self.path_to_images = path_to_images

        self._model = self.restore_model(path_to_weights_file)

    def run(self):
        """
        Model runner, processes inputs and outputing results
        """

        height, width = self._model.input_shape[1], self._model.input_shape[2]
        image_processor = ImageProcessor((height, width))

        file_names = []
        dataset = None

        # read all files and add them to arrays
        for file_name in os.listdir(self.path_to_images):
            if not (file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg")):
                continue

            image_path = os.path.join(self.path_to_images, file_name)

            if dataset is not None:
                dataset = np.append(dataset, np.array([image_processor.process_image(image_path)]), axis=0)
            else:
                dataset = np.array([image_processor.process_image(image_path)])

            file_names.append(image_path)

        # predict and get predictions as classes
        predictions = self._model.predict(dataset)
        classes = np.argmax(predictions, axis=1)

        # use stringio as we're not saving it to file
        f = StringIO()
        writer = csv.writer(f)
        for index, file_name in enumerate(file_names):
            ascii_symbol = MAPPER.class_to_ascii[classes[index]]

            text = chr(ascii_symbol)
            if text.isalpha():
                # if it's lower character then get it in uppercase
                text = text.upper()
            writer.writerow([ascii_symbol, file_name])

        # output results
        print(f.getvalue())

    @staticmethod
    def validate_path_to_images(path_to_images: str = "") -> bool:
        """
        Validates if there are any images in folder.
        We might do this on later stages, but we dont really want to load a large model without knowing if we're gonna
        use it.
        :param path_to_images: path to folder with images
        :return:
        """

        for file in os.listdir(path_to_images):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                return True
        raise FileNotFoundError(f"No images found in {path_to_images} directory")

    @staticmethod
    def restore_model(path_to_weights_file: str = ""):
        """
        Restores model from the given file
        :return:
        """
        return tf.keras.models.load_model(path_to_weights_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test task for CHI IT course")
    parser.add_argument("--input", required=True)

    args = parser.parse_args()

    program = Program("/app/model.h5", args.input)
    program.run()
