import cv2

def get_cam():
    # define a video capture object
    cam = cv2.VideoCapture(0)
    return cam