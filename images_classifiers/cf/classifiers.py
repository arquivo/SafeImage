# /home/dbicho/caffe
import os
from io import BytesIO

import numpy as np
from PIL import Image

import caffe
from classifier import Classifier


class CaffeNsfwClassifier(Classifier):

    def __init__(self):
        # TODO change this to a configuration file
        self.model = os.path.join(os.path.dirname(__file__), "models/nsfw_model/deploy.prototxt")
        self.weights = os.path.join(os.path.dirname(__file__), "models/nsfw_model/resnet_50_1by2_nsfw.caffemodel")
        self.nsfw_net = caffe.Net(self.model, self.weights, caffe.TEST)

        # reshape input data to handle batch size
        self.nsfw_net.blobs['data'].reshape(14, 3, 224, 224)

        # Load transformer
        # Note that the parameters are hard-coded for best results
        self.caffe_transformer = caffe.io.Transformer({'data': self.nsfw_net.blobs['data'].data.shape})
        self.caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
        self.caffe_transformer.set_mean('data',
                                        np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        self.caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    def resize_image(self, data, sz=(256, 256)):
        """
        Resize image. Please use this resize logic for best results instead of the
        caffe, since it was used to generate training dataset
        :param str data:
            The image data
        :param sz tuple:
            The resized image dimensions
        :returns bytearray:
            A byte array with the resized image
        """
        im = Image.open(BytesIO(data))
        if im.mode != "RGB":
            im = im.convert('RGB')
        imr = im.resize(sz, resample=Image.BILINEAR)
        fh_im = BytesIO()
        imr.save(fh_im, format='JPEG')
        fh_im.seek(0)
        return fh_im

    def caffe_preprocess_and_compute(self, pimg, output_layers=['prob']):
        """
        Run a Caffe network on an input image after preprocessing it to prepare
        it for Caffe.
        :param PIL.Image pimg:
            PIL image to be input into Caffe.
        :param caffe.Net caffe_net:
            A Caffe network with which to process pimg afrer preprocessing.
        :param list output_layers:
            A list of the names of the layers from caffe_net whose outputs are to
            to be returned.  If this is None, the default outputs for the network
            are returned.
        :return:
            Returns the requested outputs from the Caffe net.
        """
        if self.nsfw_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = self.nsfw_net.outputs

            img_bytes = self.resize_image(pimg, sz=(256, 256))
            image = caffe.io.load_image(img_bytes)

            H, W, _ = image.shape
            _, _, h, w = self.nsfw_net.blobs['data'].data.shape
            h_off = max((H - h) / 2, 0)
            w_off = max((W - w) / 2, 0)
            crop = image[int(h_off):int(h_off + h), int(w_off):int(w_off + w), :]
            transformed_image = self.caffe_transformer.preprocess('data', crop)
            transformed_image.shape = (1,) + transformed_image.shape

            input_name = self.nsfw_net.inputs[0]
            all_outputs = self.nsfw_net.forward_all(blobs=output_layers,
                                                    **{input_name: transformed_image})

            outputs = np.around(all_outputs['prob'][:, 1].astype(float), decimals=3)
            return outputs
        else:
            return []

    def caffe_batch_preprocess_and_compute(self, pimgs, output_layers=['prob']):
        """
        Run a Caffe network on an input image batch after preprocessing them to prepare
        it for Caffe.
        :param PIL.Image pimg:
            PIL image to be input into Caffe.
        :param caffe.Net caffe_net:
            A Caffe network with which to process pimg afrer preprocessing.
        :param list output_layers:
            A list of the names of the layers from caffe_net whose outputs are to
            to be returned.  If this is None, the default outputs for the network
            are returned.
        :return:
            Returns the requested outputs from the Caffe net.
        """

        if self.nsfw_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = self.nsfw_net.outputs

            transformed_images = None

            for pimg in pimgs:
                img_bytes = self.resize_image(pimg, sz=(256, 256))
                image = caffe.io.load_image(img_bytes)

                H, W, _ = image.shape
                _, _, h, w = self.nsfw_net.blobs['data'].data.shape
                h_off = max((H - h) / 2, 0)
                w_off = max((W - w) / 2, 0)
                crop = image[int(h_off):int(h_off + h), int(w_off):int(w_off + w), :]
                transformed_image = self.caffe_transformer.preprocess('data', crop)
                transformed_image.shape = (1,) + transformed_image.shape

                if transformed_images is not None:
                    # vstack transformed images
                    transformed_images = np.vstack((transformed_images, transformed_image))
                else:
                    transformed_images = transformed_image

            input_name = self.nsfw_net.inputs[0]
            all_outputs = self.nsfw_net.forward_all(blobs=output_layers, **{input_name: transformed_images})

            outputs = np.around(all_outputs['prob'][:, 1].astype(float), decimals=3)
            return outputs
        else:
            return []

    def classify_batch(self, imgs_data):
        scores = self.caffe_batch_preprocess_and_compute(imgs_data)

        # Get a list of class labels
        #with open(os.path.dirname(__file__) + "/labels.txt", 'r') as infile:
        #    class_labels = infile.readlines()[0].strip()

        result_list = []

        for score in scores:
            result_list.append(score)

        return result_list

    def classify(self, image_data):
        scores = self.caffe_preprocess_and_compute(image_data)
        # Get a list of class labels
        with open(os.path.dirname(__file__) + "/labels.txt", 'r') as infile:
            class_labels = infile.readlines()[0].strip()

        result_list = []

        for score in scores:
            result_list.append({class_labels: str(score)})

        return result_list


# TODO Transform this in a unitary test
if __name__ == '__main__':
    classifier = CaffeNsfwClassifier()
    images_path = "fffaffbfa7e84f5e81cc3469ab4bbfba.jpg", "aa4d2fb19eef48f89c01af65937c0bbb.jpg"
    images_data = []
    for image_path in images_path:
        with open(image_path, mode='rb') as fl:
            images_data.append(fl.read())

    print(classifier.classify(images_data[0]))
    print(classifier.classify(images_data[1]))
    print(classifier.classify_batch(images_data))
