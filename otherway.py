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


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = max(math.floor(normalized_x * image_width), image_width - 1)
    y_px = max(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(
    model_asset_path='model/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

cam = get_cam()

ptime = 0
while True:
    suc, frame = cam.read()

    if not suc:
        break

    img = cv2.flip(frame, 1)

    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime
    cv2.putText(img, f'FPS:{fps}', (
        img.shape[1]-120, img.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 0), 3)

    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(img, detection_result)

    cv2.imshow('Webcam', annotated_image)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
