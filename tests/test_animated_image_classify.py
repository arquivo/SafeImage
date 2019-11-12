from PIL import Image

from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier
from indexing.classify_images_index import classify_animated_image

classifier = CaffeNsfwResnetClassifier(batch_size=1, mode_gpu=False)


def test_animated_image_classification():
    with open('./test_files/problematic.gif', mode='rb') as input_file:
        img = Image.open(input_file)
        assert classify_animated_image(img, classifier) == 1.0
