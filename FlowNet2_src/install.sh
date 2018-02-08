#!/bin/bash
PYTHON=${PYTHON:-"python"}

cd ./models/components/ops/correlation
PYTHON=$PYTHON ./make.sh
cd ../resample2d
PYTHON=$PYTHON ./make.sh
cd ../channelnorm
PYTHON=$PYTHON ./make.sh
cd ../../
