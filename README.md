SafeImage
=========

SafeImage is a project that provides ways to classify images content as NSFW (Not Suitable for Work).

SafeImage is composed by two components:

1. An REST WebService API.
2. Neural Networks models to classify images
3. Classification Workers to classify images from a Redis Queue.
4. Command utility tools for testing.

Currently reported classification evaluation against the evaluation dataset:
[Resnet Model](https://github.com/arquivo/SafeImage/blob/master/docs/Resnet_NSFW_ROC.png?raw=true "ResnetNSFW ROC")
[SqueezeNet Model](https://github.com/arquivo/SafeImage/blob/master/docs/SqueezeNet_NSFW_ROC.png?raw=true "SqueezeNetNSFW ROC")


Get code from Repository
------------------------

``` sourceCode
git clone https://github.com/arquivo/SafeImage.git
```

Install Requirements
--------------------
1. Install SafeImage API:

``` sourceCode
python setup.py install
```

Launch Test Tool:
----------
``` sourceCode
cli-safeimage-test-tool --help
```

Launch Classification Workers:
------------------------------
``` sourceCode
nsfw-resnet-worker --help
nsfw-squeezenet-worker --help
```



Launch SafeImage API through uWSGI:
----------------------------------

``` sourceCode
uwsgi uwsgi.ini
```

Example of request to the API:
------------------------------
Request a POST to /safeimage path with the following JSON content:

POST /safeimage

{

  "image": image_64
  
}

Replace 'image_64' with the base64 encoded image bytes.






