import json
import time

import redis

from images_classifiers.cf.classifiers import CaffeNsfwClassifier

if __name__ == '__main__':
    db = redis.StrictRedis(
        host='localhost',
        port=6379,
    )
    classifier = CaffeNsfwClassifier()

    while True:
        # TODO change batch size
        queue = db.lrange("image_queueing", 0, 1)
        image_ids = []
        batch = []

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = q["image"]

            batch.append(image)
            image_ids.append(q["id"])

        if len(image_ids) > 0:
            print("* Batch size: {}".format(batch.shape))
            classifier.classify_batch(batch)

        time.sleep(0.25)