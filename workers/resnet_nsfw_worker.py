import base64
import json
import time

import redis
import argparse

from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier


def main():
    parser = argparse.ArgumentParser(description='Worker to consume images to be classified from a Redis Broker.')
    parser.add_argument('hostname', default='localhost', help='Specify Redis Server hostname.')
    parser.add_argument('port', default=6379, help='Specify Redis Server listening port.')
    parser.add_argument('batch_size', default=1, help='Specify the batch size to classify.')
    parser.add_argument('fetch_size', default=1, help='Specify the fetch size to get from Redis.')
    parser.add_argument('polling_time', default=0.25, help='Polling time interval in seconds.')

    args = parser.parse_args()

    init_worker(args.hostname, args.port, int(args.batch_size), int(args.fetch_size), float(args.polling_time))



def init_worker(hostname, port, batch_size, fetch_size, polling_time):
    db = redis.StrictRedis(host=hostname, port=port)

    classifier = CaffeNsfwResnetClassifier(batch_size=batch_size)

    while True:
        start = time.time()
        # queue = db.lrange("image_queueing", 0, batch_size)
        fetch_queue = db.lrange("image_queueing", 0, fetch_size)

        queues = [fetch_queue[:batch_size], fetch_queue[batch_size:]]
        for queue in queues:
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

        end = time.time()
        elapsed_time = end - start
        time.sleep(polling_time)
        # print("Elapsed Time: {}".format(elapsed_time))
        print("* Images/Second: {}".format(fetch_size / (elapsed_time + polling_time)))



if __name__ == '__main__':
    main()
