# python stress_test.py

# import the necessary packages
import base64
from threading import Thread
import requests
import time

# initialize the Keras REST API endpoint URL along with the input
# image path
REST_API_URL = "http://127.0.0.1:5000/safeimage"
IMAGE_PATH = "image_test_1.jpg"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 100
SLEEP_COUNT = 0.05


def call_predict_endpoint(n):
    # load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()

    payload = {"image": base64.b64encode(image).decode('ascii')}

    # submit the request
    # r = requests.post(REST_API_URL, files=payload).json()

    r = requests.post(REST_API_URL, json=payload).json()

    # ensure the request was sucessful
    if r["success"]:
        print("[INFO] thread {} OK Prediction: {} Image: {}".format(n, r['predictions'], IMAGE_PATH))

    # otherwise, the request failed
    else:
        print("[INFO] thread {} FAILED".format(n))


# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    # call_predict_endpoint(1)
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

# insert a long sleep so we can wait until the server is finished
# processing the images
time.sleep(300)