import argparse
import base64
import urllib2

import requests

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"


def classify_image(image_url, endpoint):
    image_64 = str(
        base64.b64encode(
            urllib2.urlopen(image_url).read()).decode(
            "ascii"))

    json_data = {"image": image_64}

    response = requests.post(endpoint, json=json_data)

    print "%s %s" % (response.content, image_url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Client to consume SafeImage API.')
    parser.add_argument('image_path', help='Specify Image URL to be classified by the API.')
    parser.add_argument('endpoint', default='http://127.0.0.1:5000/safeimage', nargs='?', help='Specify API endpoint.')

    args = parser.parse_args()

    classify_image(args.image_path, args.endpoint)
