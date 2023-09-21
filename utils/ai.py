import dlib

def get_detector():
    detector = dlib.get_frontal_face_detector()
    return detector

def get_predictor():
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    return predictor
