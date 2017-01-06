import json

from flask import Flask
from flask_autodoc import Autodoc
from flask_restful import Resource, Api
from flask_restful import reqparse

from images_classifiers.safe_image_classifier import SafeImageClassifier

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"

application = Flask(__name__)
api = Api(application)
auto = Autodoc(application)

parser = reqparse.RequestParser()
parser.add_argument('image', type=str, help='Image to be classified as safe or not')


@application.before_first_request
def init_classifier():
    """ Initiate the Classifier Object before the first request for each process."""
    global classifier
    classifier = SafeImageClassifier()

@auto.doc()
class ClassifierAPI(Resource):
    """Handles the requests to classify a image"""

    @auto.doc()
    def post(self):
        """ Handles POST request with a image to classify"""
        args = parser.parse_args()
        image_to_classify = args['image'].decode('base64')
        result = classifier.classify(image_to_classify)
        return json.dumps(result), 200


api.add_resource(ClassifierAPI, '/safeimage')

@application.route('/documentation')
def documentation():
    return auto.html()

if __name__ == '__main__':
    application.run()
