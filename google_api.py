from google.cloud import vision
from google.cloud.vision import types
import cv2
import os
import io

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/mikhail/private_key.json'
client = vision.ImageAnnotatorClient()

with io.open('./test/sprite.jpg', 'rb') as image_file:
    content = image_file.read()

image = vision.types.Image(content=content)

# Performs label detection on the image file
response = client.logo_detection(image=image)
logos = response.logo_annotations

print('Logos:')
for logo in logos:
    print(logo.description)
