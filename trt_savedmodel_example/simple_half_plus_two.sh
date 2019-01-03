#!/bin/bash
set -ex

# Prerequisites: install tensorflow with version >=1.12
WORK_DIR=/tmp/trt_saved_model_simple_half_plus_two

run_server() {
  rm -rf $WORK_DIR/*
  local saved_model_path=$WORK_DIR/saved_model
  local trt_saved_model_path=$WORK_DIR/saved_model_trt
  mkdir -p $saved_model_path $trt_saved_model_path

  curl -O \
    https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/simple_half_plus_two.py

  # TF-Serving needs the version directory, so we hard-coded version 1 here.
  python simple_half_plus_two.py     \
    --output_dir=$saved_model_path/1 \
    --output_dir_trt=$trt_saved_model_path/1

  local tag=tensorflow/serving:latest-gpu
  docker pull $tag
  docker run --runtime=nvidia -p 8500:8500 \
    --mount type=bind,source="$trt_saved_model_path",target=/models/mymodel \
    -e MODEL_NAME=mymodel -t $tag &
}

run_client() {
  curl -O \
    https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/simple_half_plus_two_client.py
  python simple_half_plus_two_client.py
}

mkdir -p $WORK_DIR
cd $WORK_DIR
mode="$1"
shift

if [[ "$mode" == "server" ]]; then
  run_server "$@"
elif [[ "$mode" == "client" ]]; then
  run_client "$@"
else
  echo "Usage: simple_half_plus_two.sh server|client"
fi
