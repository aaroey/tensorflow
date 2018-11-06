#!/bin/bash
set -ex

WORK_DIR=/tmp/trt_saved_model_generic

run_server() {
  local saved_model_url="$1"
  local trt_batch_size="$2"
  local saved_model_name="${saved_model_url##*/}"
  saved_model_name="${saved_model_name%%.tar.gz}"

  local saved_model_path=$WORK_DIR/$saved_model_name
  local trt_saved_model_path=${saved_model_path}_trt
  if [[ "$trt_batch_size" ]]; then
    rm -rf $trt_saved_model_path
  fi
  mkdir -p $saved_model_path $trt_saved_model_path

  if ! [[ -f $WORK_DIR/$saved_model_name.tar.gz ]]; then
    curl -O $saved_model_url
    tar zxf $saved_model_name.tar.gz
  fi
  local saved_model_path_to_serve=$saved_model_path

  if [[ "$trt_batch_size" ]]; then
    if ! [[ -f $trt_saved_model_path/1/saved_model.pb ]]; then
      local saved_model_path_with_version="$(echo $saved_model_path/*)"
      python <<< "
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=$trt_batch_size,
    input_saved_model_dir='$saved_model_path_with_version',
    output_saved_model_dir='$trt_saved_model_path/1')  # Hard coded version 1
"
    fi
    saved_model_path_to_serve=$trt_saved_model_path
  fi

  local tag=tensorflow/serving:nightly-gpu
  docker pull $tag
  docker run --cap-add=SYS_PTRACE --runtime=nvidia -p 8500:8500 \
    --mount type=bind,source="$saved_model_path_to_serve",target=/models/mymodel \
    -e MODEL_NAME=mymodel -t $tag &
}

mkdir -p $WORK_DIR
cd $WORK_DIR
run_server "$@"
