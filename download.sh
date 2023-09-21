#!/bin/sh

mkdir -p model
#rm -rf model/*
cd model

if [ ! -f shape_predictor_68_face_landmarks.dat ]; then
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
fi
# mediapipe's models
if [ ! -f selfie_segmenter.tflite ]; then
    wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
fi

if [ ! -f blaze_face_short_range.tflite ]; then
    wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
fi

cd ..

