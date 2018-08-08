from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier
from images_classifiers.cf.classifiers import CaffeNsfwSqueezeClassifier

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image_url')

    args = parser.parse_args()

    classifier_squeezenet = CaffeNsfwSqueezeClassifier(mode_gpu=False)
    classifier_resnet = CaffeNsfwResnetClassifier(mode_gpu=False)

    # open image
    from urllib.request import urlopen
    image_data = urlopen(args.image_url).read()

    print("SqueezeNet Result: {}".format(classifier_squeezenet.classify(image_data)))
    print("Resnet Result: {}".format(classifier_resnet.classify(image_data)))