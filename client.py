import requests
import json
import cv2

addr = 'http://localhost:105'
test_url = addr + '/detection'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('descarga.jpeg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(
    test_url, data=img_encoded.tobytes(), headers=headers)
# decode response
print(json.loads(response.text))
