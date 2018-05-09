import base64
import json
import time
import uuid

import redis
from flask import Flask
from flask import jsonify
from flask_restful import Resource, Api
from flask_restful import reqparse

from images_classifiers.cf.classifiers import CaffeNsfwClassifier

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"

# TODO make this configurable (settings.HOST etc..)
db = redis.StrictRedis(host='localhost', port=6379)

# classifier = CaffeNsfwClassifier()

application = Flask(__name__)
api = Api(application)

parser = reqparse.RequestParser()
parser.add_argument('image', type=str, help='Image to be classified as safe or not')


class ClassifierAPI(Resource):
    """Class that presents a REST API to handle image classification."""

    def post(self):
        """
        Handles POST request. The request expect a JSON document with the following structure.

            "image" : <base64> image

        The API can handle any common type of image format. (JPG, PNG, GIF, BMP, etc...)
        """
        args = parser.parse_args()

        # no need to deserialize here, it will be sent to redis first
        # image_to_classify = base64.b64decode(args['image'])

        data = {"success": False}
        key = str(uuid.uuid4())

        d = {"id": key, "image": args['image']}
        db.rpush("image_queueing", json.dumps(d))

        while True:
            output = db.get(key)

            if output is not None:
                output = output.decode("utf-8")
                data["predictions"] = json.loads(output)

                db.delete(key)
                break

            time.sleep(0.25)

        data["success"] = True

        application.logger.info('Images Classified with Score: {}'.format(data))
        return jsonify(data)


api.add_resource(ClassifierAPI, '/safeimage')

if __name__ == '__main__':
    application.run()
