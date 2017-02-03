SafeImage
=========

SafeImage is an Web Service, where the users can submit an image and obtain the NSFW (Not Safe for Work) score of the image.

SafeImage is composed by two components:

1.  An REST WebService API provided by Flask Framework.
2.  A Deep Neural Network to classify images using Tensorflow.

Currently reported evaluation against Arquivo.pt Image Search queries:

![](https://github.com/arquivo/SafeImage/blob/master/docs/ROC.png?raw=true "ROC")


Get code from Repository
------------------------

``` sourceCode
git clone http://github.com/danielbicho/safeimage.git
```

Install Requirements
--------------------

1.  An working enviroment with Python2.7.
2.  Install Requirements.txt with pip.

``` sourceCode
cd SafeImage/
pip install -r requirements.txt
```

Launch SafeImage API throug uWSGI:
----------------------------------

``` sourceCode
uwsgi uwsgi.ini
```

Test the service using the provided test client:
------------------------------------------------

``` sourceCode
python client_test.py http://example.org/image.jpg
```
