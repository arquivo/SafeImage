import json
import os
import fnmatch
import base64
import time

from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier

classifier = CaffeNsfwResnetClassifier(mode_gpu=False)

batch_size = 1

for filename in os.listdir('test_files/'):
    path = os.path.join('test_files/', filename)

    if os.path.isfile(path) and fnmatch.fnmatch(filename, 'part-*'):
        with open('indexed_test_files/{}'.format(filename), mode='a') as output:
            with open(path) as f:
                print('******* {}'.format(path))
                batch_json_docs = []
                batch_img = []

                start = time.time()
                for line in f:
                    try:
                        json_doc = json.loads(line)
                    except Exception as e:
                        print(e)
                        print(line)

                    batch_json_docs.append(json_doc)
                    batch_img.append(base64.b64decode(json_doc['imgSrcBase64']))

                    if len(batch_img) == batch_size:
                        results = classifier.classify_batch(batch_img)
                        for i in range(len(results)):
                            json_doc = batch_json_docs[i]
                            json_doc['nsfw'] = results[i]
                            output.write(json.dumps(json_doc))
                        end = time.time()
                        print('Records / Second: {}'.format(end - start))

