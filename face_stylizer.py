import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.cam import get_cam
import sys
import os
sys.path.append(os.getcwd())

cam = get_cam()

base_options = python.BaseOptions(model_asset_path='model/face_stylizer_oil_painting.task')
options = vision.FaceStylizerOptions(base_options=base_options)

stylizer = vision.FaceStylizer.create_from_options(options)

ptime = 0
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

    # Retrieve the stylized image
    stylized_image = stylizer.stylize(image)

    # Show the stylized image
    #rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)
    #resize_and_show(rgb_stylized_image)
    #output_image = img # np.where(condition, img, blurred_image)
    if stylized_image is None:
        continue
    
    output = stylized_image.numpy_view()

    cv2.putText(output, f'FPS:{fps}', (
        output.shape[1]-120, output.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 0), 3)

    # resize_and_show(output_image)
    cv2.imshow('Webcam', output)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
