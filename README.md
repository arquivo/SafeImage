# SafeImage

SafeImage is a project that provides ways to classify images content as NSFW (Not Suitable for Work).

SafeImage module provides:

1. Command utilities to classify images.
2. Classification Workers to classify images from a Redis Queue.
3. An REST WebService API.

## Resnet ROC
![resnet](https://github.com/arquivo/SafeImage/blob/master/docs/Resnet_NSFW_ROC.png?raw=true "ResnetNSFW ROC")

## SqueezeNet ROC
![squeezenet](https://github.com/arquivo/SafeImage/blob/master/docs/SqueezeNet_NSFW_ROC.png?raw=true "SqueezeNetNSFW ROC")


## Get code from Repository

``` sourceCode
git clone https://github.com/arquivo/SafeImage.git
```

## Install Requirements
#### Docker Installation

1. Clone repository: ``git clone https://github.com/arquivo/SafeImage.git``
2. Build Docker Image: ``docker build -t arquivo/safeimage``
3. Run Docker Container: ``docker run -it arquivo/safeimage <command>``

#### Install from source:
You need to  have Caffe Framework installed on your system to be able to use it.

1. Install SafeImage API:

``` sourceCode
pip install git+https://github.com/arquivo/SafeImage.git
``` 

2. Add to PYTHONPATH the directory where Caffe is installed:

``` sourceCode
export PYTHONPATH=$PYTHONPATH:/opt/caffe/python
```


Command Lines:
--------------
``` sourceCode
cli-safeimage-test-tool --help
cli-safeimage-indexing --help
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

{\
  "image": image_64  
}

Replace 'image_64' with the base64 encoded image bytes.








