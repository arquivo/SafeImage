import base64
import json
import logging
import os
import time
from io import BytesIO

import numpy as np
from PIL import Image

from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

batch_size = 1
classifier = CaffeNsfwResnetClassifier(mode_gpu=False, batch_size=batch_size)


# TODO optimize for batching
def classify_animated_image(image):
    gif_results = []

    for i in range(image.n_frames):
        image.seek(i)
        with BytesIO() as output:
            image.save(output, 'GIF')
            gif_results.append(classifier.classify(output.getvalue()))
            print("**** GIF FRAME {}: {}".format(i, gif_results[i]))

    res = np.max(np.array(results))
    print("Max result {}".format(res))
    return res


for filename in os.listdir('test_files/'):
    path = os.path.join('test_files/', filename)

    with open('indexed_test_files/nsfw_{}'.format(filename), mode='a') as output_file:
        with open(path) as input_file:

            # create a file handler
            handler = logging.FileHandler(filename + ".log")
            handler.setLevel(logging.INFO)

            # create a logging format
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)

            # add the handlers to the logger
            logger.addHandler(handler)

            logger.info('******* {}'.format(path))

            for line in input_file:
                batch_json_docs = []
                batch_img = []

                start = time.time()
                try:
                    json_doc = json.loads(line)
                except Exception as e:
                    print(e)
                    print(line)

                # verify if is an animated image
                # is_animated, actually is true for multilayers and multi frames images...
                img_bytes = base64.b64decode(json_doc['imgSrcBase64'])

                try:
                    img = Image.open(BytesIO(img_bytes))
                    if img.format == 'GIF':
                        if img.is_animated:
                            print("**** Animated GIF detected")
                            result = classify_animated_image(img)
                            json_doc['safe'] = result
                            output_file.write(json.dumps(json_doc))
                            logger.info(
                                "{} {} {} {}".format(img.format, img.n_frames, json_doc['imgSrc'], json_doc['safe']))
                            continue
                except Exception as e:
                    json_doc['safe'] = 0
                    output_file.write(json.dumps(json_doc))
                    logger.info(e)
                    logger.info("**** Problem with {}".format(json_doc['imgSrc']))
                    continue

                batch_json_docs.append(json_doc)
                batch_img.append(img_bytes)

                if len(batch_img) == batch_size:
                    results = classifier.classify_batch(batch_img)
                    for i in range(len(results)):
                        json_doc = batch_json_docs[i]
                        json_doc['safe'] = results[i]
                        output_file.write(json.dumps(json_doc))
                        logger.info("{} {}".format(json_doc['imgSrc'], json_doc['safe']))
                    end = time.time()
                    print('Records / Second: {}'.format(round(batch_size / (end - start), 2)))

            if len(batch_img) != 0:
                results = classifier.classify_batch(batch_img)
                for i in range(len(results)):
                    json_doc = batch_json_docs[i]
                    json_doc['safe'] = results[i]
                    output_file.write(json.dumps(json_doc))
                    logger.info("{} {}".format(json_doc['imgSrc'], json_doc['safe']))
