from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier
from images_classifiers.cf.classifiers import CaffeNsfwSqueezeClassifier
from six.moves.urllib.request import urlopen

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image_url')

    args = parser.parse_args()

    classifier_squeezenet = CaffeNsfwSqueezeClassifier(mode_gpu=False)
    classifier_resnet = CaffeNsfwResnetClassifier(mode_gpu=False)
    classifier_resnet_oversampling = CaffeNsfwResnetClassifier(mode_gpu=False, oversampling=True)

    # open image
    image_data = urlopen(args.image_url).read()

    squeezenet_result = classifier_squeezenet.classify(image_data)[0]
    resnet_result = classifier_resnet.classify(image_data)
    resnet_oversampling_result = classifier_resnet_oversampling.classify(image_data)

    print("SqueezeNet Result: {}".format(squeezenet_result))
    print("Resnet Result: {}".format(resnet_result))
    print("Resnet with Oversampling Result: {}".format(resnet_oversampling_result))
    print("Resnet AWP Result: {}".format(None))