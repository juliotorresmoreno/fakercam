from gestures.eyes import get_eyes
from utils.ai import get_detector, get_predictor
from gestures.lip import is_yawn, get_lip
from utils.cam import get_cam
from imutils import face_utils
import time
import cv2
import sys
import os
sys.path.append(os.getcwd())


def draw_face_rectangle(face, frame):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 00), 2)


cam = get_cam()

detector = get_detector()
predictor = get_predictor()

yawn_thresh = 35
ptime = 0
while True:
    suc, frame = cam.read()

    if not suc:
        break

    ctime = time.time()
    fps = int(1/(ctime-ptime))
    ptime = ctime
    cv2.putText(frame, f'FPS:{fps}', (
        frame.shape[1]-120, frame.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 0), 3)

    # ------Detecting face------#
    img = cv2.flip(frame, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)

    for face in faces:
        shapes = predictor(img_gray, face)
        shape = face_utils.shape_to_np(shapes)

        l1, l2 = get_lip(shape)
        e1, e2 = get_eyes(shape)
        cv2.drawContours(img, [e1, e2, l1, l2], -1,
                         (0, 165, 255), thickness=3)

        if is_yawn(shape):
            cv2.putText(
                img,
                'User Yawning!',
                (img.shape[1]//2 - 170, img.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 200), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
