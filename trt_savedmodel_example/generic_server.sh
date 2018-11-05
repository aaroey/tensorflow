#!/bin/bash
set -ex

WORK_DIR=/tmp/trt_saved_model_generic

run_server() {
  local saved_model_url="$1"
  local use_trt=''
  if [[ "$2" == 'trt' ]]; then
    use_trt=trt
  fi
  local saved_model_name="${saved_model_url##*/}"
  saved_model_name="${saved_model_name%%.tar.gz}"
  local saved_model_path=$WORK_DIR/$saved_model_name
  local trt_saved_model_path=${saved_model_path}_trt
  mkdir -p $saved_model_path $trt_saved_model_path

  if ! [[ -f $WORK_DIR/$saved_model_name.tar.gz ]]; then
    curl -O $saved_model_url
    tar zxf $saved_model_name.tar.gz
  fi
  local saved_model_path_to_serve=$saved_model_path

  if [[ "$use_trt" == 'trt' ]]; then
    if ! [[ -f $trt_saved_model_path/1/saved_model.pb ]]; then
      local saved_model_path_with_version="$(echo $saved_model_path/*)"
      python <<< "
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=128,
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
