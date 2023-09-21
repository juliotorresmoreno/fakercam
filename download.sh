#!/bin/sh

mkdir -p model
rm -rf model/*
cd model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
cd ..
