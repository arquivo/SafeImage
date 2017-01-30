import os
from collections import defaultdict
from io import BytesIO

import tensorflow as tf
from PIL import Image

from classifier import Classifier
from images_classifiers import models

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"


class NSFWClassifier(Classifier):
    def __init__(self):
        """Object constructor. Load the trained model graph to classify images."""


        # Get the data specifications for the ResNet_50_nsfw model
        self._spec = models.get_data_spec(model_class=models.ResNet_50_1by2_nsfw)

        # Create a placeholder for the input image
        self._input_node = tf.placeholder(tf.float32,
                                          shape=(None, self._spec.crop_size, self._spec.crop_size, self._spec.channels))

        # Construct the network
        self._net = models.ResNet_50_1by2_nsfw({'data': self._input_node})

        self._sess = tf.Session()

        # Load the converted parameters
        print('Loading the model')
        self._net.load(os.path.dirname(__file__) + "/nsfw.npy", self._sess)

    # TODO add support for more image types
    def load_image(self, image_data, is_jpeg):
        # Decode the image data
        img = tf.image.decode_jpeg(image_data, channels=self._spec.channels)

        if self._spec.expects_bgr:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels.
            img = tf.reverse(img, [False, False, True])
        return img

    def process_image(self, img, scale, isotropic, crop, mean):
        '''Crops, scales, and normalizes the given image.
        scale : The image wil be first scaled to this size.
                If isotropic is true, the smaller side is rescaled to this,
                preserving the aspect ratio.
        crop  : After scaling, a central crop of this size is taken.
        mean  : Subtracted from the image
        '''
        # Rescale
        if isotropic:
            img_shape = tf.to_float(tf.shape(img)[:2])
            min_length = tf.minimum(img_shape[0], img_shape[1])
            new_shape = tf.to_int32((scale / min_length) * img_shape)
        else:
            new_shape = tf.pack([scale, scale])
        img = tf.image.resize_images(img, (new_shape[0], new_shape[1]))
        # Center crop
        # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
        # See: https://github.com/tensorflow/tensorflow/issues/521
        offset = (new_shape - crop) / 2
        img = tf.slice(img, begin=tf.pack([offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
        # Mean subtraction
        return tf.to_float(img) - mean

    def classify(self, image_data):
        # Create an image producer (loads and processes images in parallel)
        img = self.load_image(image_data, True)
        # Process the image
        processed_img = self.process_image(img=img,
                                           scale=self._spec.scale_size,
                                           isotropic=self._spec.isotropic,
                                           crop=self._spec.crop_size,
                                           mean=self._spec.mean)


        # Process Image
        image_array = self._sess.run(processed_img)

        # Perform a forward pass through the network to get the class probabilities
        print('Classifying')
        predictions = self._sess.run(self._net.get_output(), feed_dict={self._input_node: [image_array]})

        # Get a list of class labels
        with open(os.path.dirname(__file__) + "/labels.txt", 'rb') as infile:
            class_labels = map(str.strip, infile.readlines())

        result = defaultdict()
        score = predictions[0]
        result[class_labels[0]] = str(score[1])

        return result


class SafeImageClassifier(Classifier):
    """Classifier Object that classify a image if its Safe or Not."""

    def __init__(self):
        """Object constructor. Load the trained model graph to classify images."""
        self.load_session_graph(os.path.dirname(__file__) + "/retrained_graph.pb")
        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
        # Loads label file, strips off carriage return
        self.label_lines = [line.rstrip() for line
                            in tf.gfile.GFile(os.path.dirname(__file__) + "/labels.txt")]

    def load_session_graph(self, graph_path):
        """ Loads the graph model."""
        with tf.gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def classify(self, image_data):
        """ Classify the image_data if its Safe or Not. Returning the score for each label."""
        image = Image.open(BytesIO(image_data))
        image_array = image.convert('RGB')
        predictions = self.sess.run(self.softmax_tensor, {'DecodeJpeg:0': image_array})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        result = defaultdict()
        for node_id in top_k:
            human_string = self.label_lines[node_id]
            score = predictions[0][node_id]
            result[human_string] = str(score)

        return result


if __name__ == '__main__':
    classifier = NSFWClassifier()
    with open('../static/images/NotSafe/5.jpg') as f:
        file_data = f.read()

    print classifier.classify(file_data)
