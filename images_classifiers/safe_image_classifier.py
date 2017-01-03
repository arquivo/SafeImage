import os
from collections import defaultdict

import tensorflow as tf

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"


class SafeImageClassifier():
    """Classifier Object that classify a image if its Safe or Not."""

    def __init__(self):
        """Object constructor. Load the trained model graph to classify images."""
        self.load_session_graph(os.path.dirname(__file__) + "/retrained_graph.pb")
        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
        # Loads label file, strips off carriage return
        self.label_lines = [line.rstrip() for line
                            in tf.gfile.GFile(os.path.dirname(__file__) + "/retrained_labels.txt")]

    def load_session_graph(self, graph_path):
        """ Loads the graph model."""
        with tf.gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def classify(self, image_data):
        """ Classify the image_data if its Safe or Not. Returning the score for each label."""
        predictions = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        result = defaultdict()
        for node_id in top_k:
            human_string = self.label_lines[node_id]
            score = predictions[0][node_id]
            result[human_string] = str(score)

        return result
