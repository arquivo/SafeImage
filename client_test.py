import base64

import requests

image_64 = str(
    base64.b64encode(open('/home/dbicho/Documents/FilterProject/validation_images/ana-malhoa.jpg', "rb").read()).decode(
        "ascii"))

url = 'http://127.0.0.1:8080/safeimage'

json_data = {"image": image_64}

response = requests.post(url, json=json_data)

print response.content
