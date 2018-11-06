import base64
import json
import time
import uuid
from threading import Thread

import redis


def rpush_redis(db, img_data):
    key = str(uuid.uuid4())

    d = {"id": key, "image": img_data}
    db.rpush("image_queueing", json.dumps(d))


db = redis.StrictRedis(
    host='193.136.192.183',
    port=6379,
)

with open('image_test_1.jpg', mode='rb') as input_handler:
    img_data = base64.b64encode(input_handler.read()).decode("ascii")

    NUM_REQUESTS = 1000
    SLEEP_COUNT = 0.05

    # loop over the number of threads
    for i in range(0, NUM_REQUESTS):
        print("Thread fire..")
        t = Thread(target=rpush_redis, args=(db, img_data,))
        t.daemon = True
        t.start()
        time.sleep(SLEEP_COUNT)

# insert a long sleep so we can wait until the server is finished
# processing the images
time.sleep(300)
