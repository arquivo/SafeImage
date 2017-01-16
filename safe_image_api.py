import json
import os

from flask import Flask
from flask import render_template
from flask_restful import Resource, Api
from flask_restful import reqparse

from images_classifiers.safe_image_classifier import SafeImageClassifier

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"

application = Flask(__name__)
api = Api(application)

parser = reqparse.RequestParser()
parser.add_argument('image', type=str, help='Image to be classified as safe or not')


@application.before_first_request
def init_classifier():
    """ Initiate the Classifier Object before the first request for each process."""
    global classifier
    classifier = SafeImageClassifier()

class ClassifierAPI(Resource):
    """Class that presents a REST API to handle image classification."""

    def post(self):
        """
        Handles POST request. The request expect a json document with the following structure:

        Endpoint URL: <host>/safeimage
        Request Body:
        {
            "image" : <base64> image
        }

        The API can handle any common type of image format. (JPG, PNG, GIF, BMP, etc...)
        """
        args = parser.parse_args()
        image_to_classify = args['image'].decode('base64')
        result = classifier.classify(image_to_classify)
        return json.dumps(result), 200

@application.route('/backend')
def testing_backend():
    """Debug Endpoint to image classifaction. Classify images at the folder /static/images."""
    scores = []
    for path in os.listdir('./static/images/'):
        test_image_path =os.path.join('./static/images/',path)
        with open(test_image_path, mode='rb') as f:
            scores.append([path, classifier.classify(f.read())])
    return render_template('results_view.html', scores = scores )

api.add_resource(ClassifierAPI, '/safeimage')


if __name__ == '__main__':
    application.run()
