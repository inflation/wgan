import os
import re

import tensorflow as tf


class ImageDataset:
    def __init__(self, data_path, batch_size=128, num_preprocess_threads=4,
                 output_height=128, output_width=128, channels=3):

        self.batch_size = batch_size
        self.num_preprocess_threads = num_preprocess_threads
        self.output_height = output_height
        self.output_width = output_width
        self.channels = channels

        if not os.path.isdir(data_path):
            raise AttributeError("The path is not a directory!")

        filenames = [f for f in os.listdir(data_path) if re.search(r'.*\.(jpg|png)', f)]

        if len(filenames) == 0:
            raise AttributeError("The path does not contains any images!")

        self.reader = tf.WholeFileReader(name="image_reader")
        self.producer = tf.train.string_input_producer([os.path.join(data_path, name) for name in filenames],
                                                       capacity=4 * batch_size)
        self.batch = tf.train.shuffle_batch(
                        [self.get_image()],
                        batch_size=self.batch_size,
                        num_threads=self.num_preprocess_threads,
                        capacity=4 * self.batch_size,
                        min_after_dequeue=self.batch_size,
                        name='get_batch')

    def get_image(self):
        _, image = self.reader.read(self.producer)
        image = tf.image.decode_image(image)
        image = tf.image.resize_image_with_crop_or_pad(image, self.output_height, self.output_width)
        image.set_shape([self.output_height, self.output_width, self.channels])

        return image

    def get_batch(self):
        return self.batch


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    with tf.Session() as sess:
        image_dataset = ImageDataset("/Users/inflation/Pictures/faces",
                                     batch_size=16,
                                     num_preprocess_threads=4)

        coord = tf.train.Coordinator()

        batch = image_dataset.get_batch()

        threads = tf.train.start_queue_runners(coord=coord)

        for idx, image in enumerate(batch.eval()):
            plt.subplot(4, 4, 1+idx)
            plt.imshow(image)

        plt.figure()
        for idx, image in enumerate(batch.eval()):
            plt.subplot(4, 4, 1+idx)
            plt.imshow(image)

        plt.show()

        coord.request_stop()
        coord.join(threads)

