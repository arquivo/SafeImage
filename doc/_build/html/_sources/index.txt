.. SafeImage documentation master file, created by
   sphinx-quickstart on Mon Jan 16 21:28:32 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SafeImage
=====================================

SafeImage is an Web API that the users submit an Image and the service classify the image if it has explicit content or not.

SafeImage is composed with two components:

#. An REST WebService API provided by Flask Framework.
#. A Deep Neural Network to classify images using Tensorflow.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 2

   images_classifiers
   images_classifiers.models
   safe_image_api
   client_test


