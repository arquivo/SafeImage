import json

from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse

from images_classifiers.safe_image_classifier import SafeImageClassifier

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"

application = Flask(__name__)
api = Api(application)

parser = reqparse.RequestParser()
parser.add_argument('image', type=str, help='Image to be classified as safe or not')

classifier = SafeImageClassifier()


class ClassifierAPI(Resource):
    def post(self):
        args = parser.parse_args()
        image_to_classify = args['image'].decode('base64')
        result = classifier.classify(image_to_classify)
        return json.dumps(result), 200

    def get(self):
        return 200

api.add_resource(ClassifierAPI, '/safeimage')

if __name__ == '__main__':
    application.run()

