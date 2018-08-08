import os
import time
from collections import defaultdict
from io import BytesIO

import numpy as np
from PIL import Image
from PIL import ImageFile

import caffe
from ..classifier import Classifier

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(data, sz=(256, 256)):
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


def load_synset(path):
    labels = defaultdict()
    with open(path) as input:
        index_incr = 0
        for line in input.readlines():
            labels[index_incr] = line.strip()
            index_incr += 1
    return labels


class CaffeImageNetVGG19Classifier(Classifier):

    def __init__(self, batch_size=1, mode_gpu=True):
        self.synnet_labels = load_synset(
            os.path.join(os.path.dirname(__file__), "models/vgg19_imagenet_model/synset.txt"))

        self.mode_gpu = mode_gpu
        self.model = os.path.join(os.path.dirname(__file__), "models/vgg19_imagenet_model/deploy.prototxt")
        self.weights = os.path.join(os.path.dirname(__file__),
                                    "models/vgg19_imagenet_model/VGG_ILSVRC_19_layers.caffemodel")

        self.vgg19_imagenet_net = caffe.Net(self.model, self.weights, caffe.TEST)

        # reshape input data to handle batch size
        self.reshape = self.vgg19_imagenet_net.blobs['data'].reshape(batch_size, 3, 224, 224)

        # Load transformer
        # Note that the parameters are hard-coded for best results
        self.caffe_transformer = caffe.io.Transformer({'data': self.vgg19_imagenet_net.blobs['data'].data.shape})
        self.caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
        self.caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.caffe_transformer.set_mean('data', np.array(
            [103.939, 116.779, 123.68]))  # subtract the dataset-mean value in each channel
        self.caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

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
        if self.mode_gpu:
            caffe.set_mode_gpu()

        if self.vgg19_imagenet_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = self.vgg19_imagenet_net.outputs

            img_bytes = resize_image(pimg, sz=(256, 256))
            image = caffe.io.load_image(img_bytes)

            H, W, _ = image.shape
            _, _, h, w = self.vgg19_imagenet_net.blobs['data'].data.shape
            h_off = max((H - h) / 2, 0)
            w_off = max((W - w) / 2, 0)
            crop = image[int(h_off):int(h_off + h), int(w_off):int(w_off + w), :]
            transformed_image = self.caffe_transformer.preprocess('data', crop)
            transformed_image.shape = (1,) + transformed_image.shape

            input_name = self.vgg19_imagenet_net.inputs[0]
            all_outputs = self.vgg19_imagenet_net.forward_all(blobs=output_layers, **{input_name: transformed_image})

            label_indexes = np.argsort(all_outputs['prob']).flatten()[::-1][:5]

            outputs = [self.synnet_labels[index] for index in label_indexes]
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
        if self.mode_gpu:
            caffe.set_mode_gpu()

        if self.vgg19_imagenet_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = self.vgg19_imagenet_net.outputs

            transformed_images = None

            start = time.time()
            for pimg in pimgs:
                img_bytes = resize_image(pimg, sz=(256, 256))
                image = caffe.io.load_image(img_bytes)

                H, W, _ = image.shape
                _, _, h, w = self.vgg19_imagenet_net.blobs['data'].data.shape
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
            end = time.time()
            input_name = self.vgg19_imagenet_net.inputs[0]
            all_outputs = self.vgg19_imagenet_net.forward_all(blobs=output_layers, **{input_name: transformed_images})

            outputs = []

            for i in range(all_outputs['prob'].shape[0]):
                label_indexes = np.argsort(all_outputs['prob']).flatten()[::-1][:5]
                outputs.append([self.synnet_labels[index] for index in label_indexes])

            print("Time Preprocessing Images: {} seconds".format(end - start))
            return outputs
        else:
            return []

    def classify_batch(self, imgs_data):
        labels = self.caffe_batch_preprocess_and_compute(imgs_data)
        return labels

    def classify(self, image_data):
        labels = self.caffe_preprocess_and_compute(image_data)
        return labels


class CaffeNsfwSqueezeClassifier(Classifier):

    def __init__(self, batch_size=1, mode_gpu=True):
        self.mode_gpu = mode_gpu
        self.model = os.path.join(os.path.dirname(__file__), "models/nsfw_squeezenet_model/deploy.prototxt")
        self.weights = os.path.join(os.path.dirname(__file__),
                                    "models/nsfw_squeezenet_model/nsfw_squeezenet.caffemodel")
        self.squeeze_nsfw_net = caffe.Net(self.model, self.weights, caffe.TEST)

        # reshape input data to handle batch size
        self.reshape = self.squeeze_nsfw_net.blobs['data'].reshape(batch_size, 3, 227, 227)

        # Load transformer
        # Note that the parameters are hard-coded for best results
        self.caffe_transformer = caffe.io.Transformer({'data': self.squeeze_nsfw_net.blobs['data'].data.shape})
        self.caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
        self.caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.caffe_transformer.set_mean('data', np.array(
            [112.005, 120.294, 138.682]))  # subtract the dataset-mean value in each channel
        self.caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

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
        if self.mode_gpu:
            caffe.set_mode_gpu()

        if self.squeeze_nsfw_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = self.squeeze_nsfw_net.outputs

            img_bytes = resize_image(pimg, sz=(256, 256))
            image = caffe.io.load_image(img_bytes)

            H, W, _ = image.shape
            _, _, h, w = self.squeeze_nsfw_net.blobs['data'].data.shape
            h_off = max((H - h) / 2, 0)
            w_off = max((W - w) / 2, 0)
            crop = image[int(h_off):int(h_off + h), int(w_off):int(w_off + w), :]
            transformed_image = self.caffe_transformer.preprocess('data', crop)
            transformed_image.shape = (1,) + transformed_image.shape

            input_name = self.squeeze_nsfw_net.inputs[0]
            all_outputs = self.squeeze_nsfw_net.forward_all(blobs=output_layers, **{input_name: transformed_image})

            outputs = np.around(all_outputs['prob'][:, 1].flatten().astype(float), decimals=3)
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
        if self.mode_gpu:
            caffe.set_mode_gpu()

        if self.squeeze_nsfw_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = self.squeeze_nsfw_net.outputs

            transformed_images = None

            start = time.time()
            for pimg in pimgs:
                img_bytes = resize_image(pimg, sz=(256, 256))
                image = caffe.io.load_image(img_bytes)

                H, W, _ = image.shape
                _, _, h, w = self.squeeze_nsfw_net.blobs['data'].data.shape
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
            end = time.time()
            input_name = self.squeeze_nsfw_net.inputs[0]
            all_outputs = self.squeeze_nsfw_net.forward_all(blobs=output_layers, **{input_name: transformed_images})

            outputs = np.around(all_outputs['prob'][:, 1].flatten().astype(float), decimals=3)

            print("Time Preprocessing Images: {} seconds".format(end - start))
            return outputs
        else:
            return []

    def classify_batch(self, images_data):
        scores = self.caffe_batch_preprocess_and_compute(images_data)
        return scores.tolist()

    def classify(self, image_data):
        scores = self.caffe_preprocess_and_compute(image_data)
        return scores.tolist()


class CaffeNsfwResnetClassifier(Classifier):

    def __init__(self, batch_size=1, mode_gpu=True, oversampling=False):
        self.oversmapling = oversampling
        self.mode_gpu = mode_gpu
        self.model = os.path.join(os.path.dirname(__file__), "models/nsfw_resnet_model/deploy.prototxt")
        self.weights = os.path.join(os.path.dirname(__file__),
                                    "models/nsfw_resnet_model/resnet_50_1by2_nsfw.caffemodel")
        self.nsfw_net = caffe.Net(self.model, self.weights, caffe.TEST)

        # reshape input data to handle batch size
        self.reshape = self.nsfw_net.blobs['data'].reshape(batch_size, 3, 224, 224)

        # Load transformer
        # Note that the parameters are hard-coded for best results
        self.caffe_transformer = caffe.io.Transformer({'data': self.nsfw_net.blobs['data'].data.shape})
        self.caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
        self.caffe_transformer.set_mean('data',
                                        np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        self.caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

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

        if self.mode_gpu:
            caffe.set_mode_gpu()

        sz = (224, 244) if not self.oversmapling else (256, 256)

        if self.nsfw_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = self.nsfw_net.outputs

            img_bytes = resize_image(pimg, sz=sz)
            image = caffe.io.load_image(img_bytes)

            if self.oversmapling:
                transformed_images = None
                oversampled_images = caffe.io.oversample([image], (224, 224))

                for oversampled_image in oversampled_images:
                    transformed_image = self.caffe_transformer.preprocess('data', oversampled_image)
                    transformed_image.shape = (1,) + transformed_image.shape

                if transformed_images is not None:
                    # vstack transformed images
                    transformed_images = np.vstack((transformed_images, transformed_image))
                else:
                    transformed_images = transformed_image
            else:
                transformed_images = self.caffe_transformer.preprocess('data', image)
                transformed_images.shape = (1,) + transformed_images.shape

            input_name = self.nsfw_net.inputs[0]
            all_outputs = self.nsfw_net.forward_all(blobs=output_layers,
                                                    **{input_name: transformed_images})

            outputs = all_outputs['prob'][:, 1].astype(float)
            return np.around(np.mean(outputs), decimals=3)
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

        if self.mode_gpu:
            caffe.set_mode_gpu()

        if not self.oversmapling:
            sz = (224, 224)
        else:
            sz = (256, 256)

        if self.nsfw_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = self.nsfw_net.outputs

            transformed_images = None

            for pimg in pimgs:
                try:
                    img_bytes = resize_image(pimg, sz=sz)

                    image = caffe.io.load_image(img_bytes)

                    transformed_image = self.caffe_transformer.preprocess('data', image)
                    transformed_image.shape = (1,) + transformed_image.shape
                except Exception as e:
                    # TODO get a better method to handle this.
                    # if error classify a blank image
                    transformed_image = np.zeros((1, 3, 224, 224))
                    print(e)

                if self.oversmapling:
                    oversampled_images = caffe.io.oversample([image], (224, 224))

                    for oversampled_image in oversampled_images:
                        transformed_image = self.caffe_transformer.preprocess('data', oversampled_image)
                        transformed_image.shape = (1,) + transformed_image.shape

                        if transformed_images is not None:
                            # vstack transformed images
                            transformed_images = np.vstack((transformed_images, transformed_image))
                        else:
                            transformed_images = transformed_image
                else:
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
        return scores.tolist()

    def classify(self, image_data):
        scores = self.caffe_preprocess_and_compute(image_data)
        return scores.tolist()


class CaffeImageNetInception21K(Classifier):
    pass
