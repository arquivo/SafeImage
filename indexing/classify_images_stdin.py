import argparse
import base64
import json
import sys
from io import BytesIO

import numpy as np
from PIL import Image

from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier


# TODO optimize for batching
def classify_animated_image(image, classifier):
    gif_results = []

    for i in range(image.n_frames):
        image.seek(i)
        with BytesIO() as output:
            image.save(output, 'GIF')
            gif_results.append(classifier.classify(output.getvalue()))
    res = np.max(np.array(gif_results))
    return res


def main():
    parser = argparse.ArgumentParser(description='Classify Images from Arquivo.pt Image Index JSON.')
    parser.add_argument('--batch_size', default=1, help='Specify the batch size to classify.')
    parser.add_argument('--mode_gpu', action='store_true', help='Specify GPU or CPU index classification.')

    args = parser.parse_args()

    batch_size = int(args.batch_size)

    mode_gpu = True if args.mode_gpu else False

    classifier = CaffeNsfwResnetClassifier(mode_gpu=mode_gpu, batch_size=batch_size)

    # create a file handler
    batch_json_docs = []
    batch_img = []

    for line in input():
        try:
            json_doc = json.loads(line)  # verify if is an animated image
            # is_animated, actually is true for multilayers and multi frames images...
            img_bytes = base64.b64decode(json_doc['imgSrcBase64'])

            try:
                img = Image.open(BytesIO(img_bytes))
                if img.format == 'GIF':
                    if img.is_animated:
                        result = classify_animated_image(img, classifier)
                        json_doc['safe'] = result
                        json_doc['animated'] = True
                        print(json.dumps(json_doc))
                        continue
            except Exception as e:
                json_doc['safe'] = 0
                print(json.dumps(json_doc))
                continue

            batch_json_docs.append(json_doc)
            batch_img.append(img_bytes)

            if len(batch_img) == batch_size:
                results = classifier.classify_batch(batch_img)
                for i in range(len(results)):
                    json_doc = batch_json_docs[i]
                    json_doc['safe'] = results[i]
                    json_doc['animated'] = False
                    print(json.dumps(json_doc))

                batch_json_docs = []
                batch_img = []

        except Exception as e:
            sys.stderr.write("Error ocurred: {} at line {}".format(e, line))

        if len(batch_img) != 0:
            results = classifier.classify_batch(batch_img)
            for i in range(len(results)):
                json_doc = batch_json_docs[i]
                json_doc['safe'] = results[i]
                json_doc['animated'] = False
                print(json.dumps(json_doc))


if __name__ == '__main__':
    main()
