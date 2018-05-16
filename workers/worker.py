import base64
import json
import time

import yaml
import redis

from images_classifiers.cf.classifiers import CaffeNsfwClassifier

if __name__ == '__main__':
    with open('config.yaml', mode='r') as cf:
        config = yaml.load(cf)

    db = redis.StrictRedis(
        host=config['redis_hostname'],
        port=config['redis_port'],
    )

    classifier = CaffeNsfwClassifier()

    while True:
        # TODO change batch size
        queue = db.lrange("image_queueing", 0, 8)
        image_ids = []
        batch = []

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = q["image"]

            batch.append(base64.b64decode(image))
            image_ids.append(q["id"])

        if len(image_ids) > 0:
            print("* Batch size: {}".format(len(batch)))
            results = classifier.classify_batch(batch)

            for i in range(len(image_ids)):
                db.set(image_ids[i], json.dumps(results[i]))

            db.ltrim("image_queueing", len(image_ids), -1)

        time.sleep(0.25)
