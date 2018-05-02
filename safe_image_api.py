import os

import numpy as np
from flask import Flask
from flask import jsonify
from flask import render_template
from flask_restful import Resource, Api
from flask_restful import reqparse
from sklearn import metrics
import base64

from images_classifiers.cf.classifiers import CaffeNsfwClassifier

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"


classifier = CaffeNsfwClassifier()

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
        image_to_classify = base64.b64decode(args['image'])
        result = classifier.classify(image_to_classify)
        application.logger.info('Images Classified with Score: {}'.format(result))
        return jsonify(result)


@application.route('/backend')
def testing_backend():
    """Debug Endpoint to image classifaction. Classify images at the folder /static/images."""
    scores_safe = []
    for path in os.listdir('./static/images/Safe'):
        test_image_path = os.path.join('./static/images/Safe', path)
        with open(test_image_path, mode='rb') as f:
            scores_safe.append([path, classifier.classify(f.read())])

    scores_not_safe = []
    for path in os.listdir('./static/images/NotSafe'):
        test_image_path = os.path.join('./static/images/NotSafe', path)
        with open(test_image_path, mode='rb') as f:
            scores_not_safe.append([path, classifier.classify(f.read())])

    y_ = []

    for result in scores_safe:
        if float(result[1]['NSFW']) <= 0.7:
            y_.append(0)
        else:
            y_.append(1)

    for result in scores_not_safe:
        if float(result[1]['NSFW']) > 0.7:
            y_.append(1)
        else:
            y_.append(0)

    y_score = np.asarray(y_)
    y_label = np.hstack((np.zeros(len(scores_safe)), (np.ones(len(scores_not_safe)))))

    # report = metrics.classification_report(y_label, y_score, target_names=['Safe', 'Not Safe'])

    # precision recall f1-score support Safe 0.91 0.80 0.85 50 Not Safe 0.76 0.89 0.82 35 avg / total 0.85 0.84 0.84 85
    AUC = metrics.roc_auc_score(y_label, y_score)
    ACC = metrics.accuracy_score(y_label, y_score)
    CF = metrics.confusion_matrix(y_label, y_score)

    return render_template('results_view.html', scores_safe=scores_safe, scores_not_safe=scores_not_safe, AUC=AUC,
                           ACC=ACC, CF=CF)


api.add_resource(ClassifierAPI, '/safeimage')

if __name__ == '__main__':
    application.run()
