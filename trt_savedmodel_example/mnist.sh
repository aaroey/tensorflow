#!/bin/bash
set -ex

# Prerequisites: install tensorflow with version >=1.12
WORK_DIR=/tmp/trt_saved_model_mnist

run_server() {
  local saved_model_path=$WORK_DIR/saved_model
  local trt_saved_model_path=${saved_model_path}_trt
  local url_prefix=https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example
  curl -O $url_prefix/mnist_saved_model.py
  curl -O $url_prefix/mnist_input_data.py

  python mnist_saved_model.py \
    --training_iteration=100  \
    --model_version=1         \
    --work_dir=$WORK_DIR      \
    $saved_model_path

  #if ! [[ -f $trt_saved_model_path/1/saved_model.pb ]]; then
    python <<< "
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=1,
    input_saved_model_dir='$saved_model_path/1',
    output_saved_model_dir='$trt_saved_model_path/1')  # Hard coded version 1
"
  #fi
  return

  local tag=''
  if [[ "$1" == 'nightly' ]]; then
    tag=tensorflow/serving:nightly-gpu
    docker pull $tag
  else
    tag=tensorflow/serving:latest-gpu
    docker pull $tag
  fi

  docker run --runtime=nvidia -p 8500:8500 \
    --mount type=bind,source="$trt_saved_model_path",target=/models/mymodel \
    -e MODEL_NAME=mymodel -t $tag &
}

run_client() {
  curl -O https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/mnist_client.py
  python resnet50_client.py \
    --num_tests=10 \
    --server=127.0.0.1:8500
}

mkdir -p $WORK_DIR
cd $WORK_DIR
mode="$1"
shift

if [[ "$mode" == "server" ]]; then
  run_server "$@"
elif [[ "$mode" == "client" ]]; then
  run_client "$@"
elif [[ "$mode" == "clear" ]]; then
  rm -rf $WORK_DIR
else
  echo "Usage: mnist.sh server|client|clean"
fi
