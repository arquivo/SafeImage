.. SafeImage documentation master file, created by
   sphinx-quickstart on Mon Jan 16 21:28:32 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SafeImage
=====================================

SafeImage is an Web Service, where the users can submit an image and obtain the classifcation result if it has explicit content or not.

SafeImage is composed with two components:

#. An REST WebService API provided by Flask Framework.
#. A Deep Neural Network to classify images using Tensorflow.

Get code from Repository
------------------------

.. code-block:: html
    :linenos:

    git clone http://github.com/danielbicho/safeimage.git


Install Requirements
--------------------

#. An working enviroment with Python2.7.
#. Install Requirements.txt with pip.

.. code-block:: html
    :linenos:

    cd SafeImage/
    pip install -r requirements.txt

Launch SafeImage API throug uWSGI:
----------------------------------
.. code-block:: html
    :linenos:

    uwsgi uwsgi.ini

Test the service using the provided test client:
------------------------------------------------
.. code-block:: html
    :linenos:

    python client_test.py http://example.org/image.jpg


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


