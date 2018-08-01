import argparse
import base64
from urllib.request import urlopen

import requests

__author__ = "Daniel Bicho"
__email__ = "daniel.bicho@fccn.pt"


def classify_image(image_url, endpoint):
    """Grab a image and classify it through a Classifier Service Endpoint.

    Args:
        image_url: An image URL, the image can be at a remote location (http://) or at the file system (file://)
        endpoint: The URL for the Service Endpoint (example: http://127.0.0.1:5000/safeimage)

    Returns:
        The response JSON string with the classification score to the system output.
    """

    # Encode with base64 the image bits to send to through the wire
    image_64 = base64.b64encode(
            urlopen(image_url).read()).decode(
            "ascii")

    json_data = {"image": image_64}

    response = requests.post(endpoint, json=json_data)

    print("%s %s" % (response.content, image_url))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Client to consume SafeImage API.')
    parser.add_argument('image_path', help='Specify Image URL to be classified by the API.')
    parser.add_argument('endpoint', default='http://127.0.0.1:5000/safeimage', nargs='?', help='Specify API endpoint.')

    args = parser.parse_args()

    classify_image(args.image_path, args.endpoint)
