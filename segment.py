import time
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.cam import get_cam
import sys
import os
sys.path.append(os.getcwd())


# STEP 2: Create an FaceDetector object.
# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='model/selfie_segmenter.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

cam = get_cam()

background_path = 'assets/hogwarts_legacy-wallpaper.jpg'
background = cv2.imread(background_path)

ptime = 0

# Create the segmenter
segmenter = python.vision.ImageSegmenter.create_from_options(options)

while True:
    suc, frame = cam.read()

    if not suc:
        break

    img = cv2.flip(frame, 1)

    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime

    # Create the MediaPipe Image
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    # Retrieve the category masks for the image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    # Convert the BGR image to RGB
    #image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)
    #image_data[:][:] = sum(image_data[:][:] * .7)

    # Apply effects
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(background, (55, 55), 3)

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) < 0.1
    output_image = np.where(condition, img, blurred_image)

    cv2.putText(output_image, f'FPS:{fps}', (
        output_image.shape[1]-120, output_image.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 0), 3)

    # resize_and_show(output_image)
    cv2.imshow('Webcam', output_image)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
