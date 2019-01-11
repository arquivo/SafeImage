import argparse
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


# TODO optimize for batching
def classify_animated_image(image, classifier):
    gif_results = []

    for i in range(image.n_frames):
        image.seek(i)
        with BytesIO() as output:
            image.save(output, 'GIF')
            gif_results.append(classifier.classify(output.getvalue()))
            logger.debug("**** GIF FRAME {}: {}".format(i, gif_results[i]))

    res = np.max(np.array(gif_results))
    logger.debug("Max result {}".format(res))

    return res


def main():
    parser = argparse.ArgumentParser(description='Classify Images from Arquivo.pt Image Index JSON.')
    parser.add_argument('input_folder', default='indexes', help='Specify input folder location with Image Indexes.')
    parser.add_argument('--batch_size', default=1, help='Specify the batch size to classify.')
    parser.add_argument('--mode_gpu', action='store_true', help='Specify GPU or CPU index classification.')
    parser.add_argument('--debug', action='store_true', help='Set debug mode.')

    args = parser.parse_args()

    batch_size = args.batch_size

    mode_gpu = True if args.mode_gpu else False

    classifier = CaffeNsfwResnetClassifier(mode_gpu=mode_gpu, batch_size=batch_size)

    for filename in os.listdir(args.input_folder):
        path = os.path.join(args.input_folder, filename)

        with open('{}/nsfw_{}'.format(args.input_folder, filename), mode='a') as output_file:
            with open(path) as input_file:

                # create a file handler
                handler = logging.FileHandler(filename + ".log")

                if args.debug:
                    handler.setLevel(logging.DEBUG)
                else:
                    handler.setLevel(logging.INFO)

                # create a logging format
                formatter = logging.Formatter('%(message)s')
                handler.setFormatter(formatter)

                # add the handlers to the logger
                logger.addHandler(handler)

                logger.info('******* Classifying Index {}'.format(path))

                for line in input_file:
                    batch_json_docs = []
                    batch_img = []

                    start = time.time()
                    try:
                        json_doc = json.loads(line)
                    except Exception as e:
                        logger.info("Error ocurred: {}".format(e))
                        logger.info("At line {}".format(line))

                    # verify if is an animated image
                    # is_animated, actually is true for multilayers and multi frames images...
                    img_bytes = base64.b64decode(json_doc['imgSrcBase64'])

                    try:
                        img = Image.open(BytesIO(img_bytes))
                        if img.format == 'GIF':
                            if img.is_animated:
                                print("**** Animated GIF detected")
                                result = classify_animated_image(img, classifier)
                                json_doc['safe'] = result
                                output_file.write(json.dumps(json_doc))
                                logger.info(
                                    "{} {} {} {}".format(img.format, img.n_frames, json_doc['imgSrc'],
                                                         json_doc['safe']))
                                continue
                    except Exception as e:
                        json_doc['safe'] = 0
                        output_file.write(json.dumps(json_doc))
                        logger.info("Error Ocurred at {}".format(e))
                        logger.info("Problem with {}".format(json_doc['imgSrc']))
                        continue

                    batch_json_docs.append(json_doc)
                    batch_img.append(img_bytes)

                    if len(batch_img) == batch_size:
                        results = classifier.classify_batch(batch_img)
                        for i in range(len(results)):
                            json_doc = batch_json_docs[i]
                            json_doc['safe'] = results[i]
                            output_file.write(json.dumps(json_doc))
                            logger.info("{} {} {}".format(json_doc['timestamp'], json_doc['imgSrc'], json_doc['safe']))
                        end = time.time()
                        print('Records / Second: {}'.format(round(batch_size / (end - start), 2)))

                if len(batch_img) != 0:
                    results = classifier.classify_batch(batch_img)
                    for i in range(len(results)):
                        json_doc = batch_json_docs[i]
                        json_doc['safe'] = results[i]
                        output_file.write(json.dumps(json_doc))
                        logger.info("{} {} {}".format(json_doc['timestamp'], json_doc['imgSrc'], json_doc['safe']))


if __name__ == '__main__':
    main()
