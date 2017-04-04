import json
import sys

import requests

sys.path.append("..")

from images_classifiers.classifiers import NSFWClassifier

def nsfw_index(path):
    classifier = NSFWClassifier()
    count = 0
    with open('indexed.jsonl', mode='wa') as output:
        with open(path) as f:
            for line in f:
                json_doc = json.loads(line)
                score = classifier.classify(json_doc['srcBase64'].decode('base64'))
                json_doc['NSFW'] = score['NSFW']
                json_str = json.dumps(json_doc)
                output.write(json_str + '\n')
                count = count + 1
                print count

def nsfw_index_sevice(path, endpoint):
    count = 0
    with open('indexed.jsonl', mode='wa') as output:
        with open(path) as f:
            for line in f:
                json_doc = json.loads(line)
                json_data = {"image": json_doc['srcBase64']}
                response = requests.post(endpoint, json=json_data)
                json_doc['NSFW'] = json.loads(response.content)['NSFW']

                json_str = json.dumps(json_doc)
                output.write(json_str + '\n')
                count = count + 1
                print count


if __name__ == '__main__':
    #nsfw_index('teste1.jsonl')
    nsfw_index_sevice('teste1.jsonl', 'http://p27.arquivo.pt:9080/safeimage')