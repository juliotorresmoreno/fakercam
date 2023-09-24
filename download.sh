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

if [ ! -f deeplab_v3.tflite ]; then
    wget https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite
fi

if [ ! -f hand_landmarker.task ]; then
    wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
fi

if [ ! -f hair_segmenter.tflite ]; then
    wget https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite
fi

if [ ! -f face_stylizer_color_sketch.task ]; then
    wget https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_color_sketch.task
fi

if [ ! -f face_stylizer_color_ink.task ]; then
    wget https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_color_ink.task
fi

if [ ! -f face_stylizer_oil_painting.task ]; then
    wget https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_oil_painting.task
fi

if [ ! tensorflow_inception_graph.pb ];then
    wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    unzip inception5h.zip
fi

cd ..

