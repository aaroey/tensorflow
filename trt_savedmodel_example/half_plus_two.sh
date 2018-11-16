#!/bin/bash
set -ex

# Prerequisites: install tensorflow with version >=1.13
WORK_DIR=/tmp/trt_saved_model_half_plus_two

run_server() {
  local saved_model_path=$WORK_DIR/saved_model
  local trt_saved_model_path=$WORK_DIR/saved_model_trt

  local run_options="--rm --runtime=nvidia -it --network=host -v $(pwd):$(pwd) -v /tmp:/tmp"

  local devel_image=tensorflow/serving:$1-devel-gpu
  docker pull $devel_image
  docker run $run_options $devel_image bash -c "
    rm -rf $saved_model_path* $trt_saved_model_path
    cd $(pwd)
    pip install tensorflow-gpu

    curl -O \
      https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/saved_model/saved_model_half_plus_two.py

    python saved_model_half_plus_two.py              \
      --device=gpu                                   \
      --output_dir=$saved_model_path/1               \
      --output_dir_pbtxt=${saved_model_path}_pbtxt/1 \
      --output_dir_main_op=${saved_model_path}_main_op/1

    python -c '
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=1,
    input_saved_model_dir=\"$saved_model_path/1\",
    output_saved_model_dir=\"$trt_saved_model_path/1\")  # Hard coded version 1
'
  "

  local saved_model_path_to_serve="$saved_model_path"
  if [[ "${USE_TRT:-true}" == 'true' ]]; then
    saved_model_path_to_serve="$trt_saved_model_path"
  fi

  local image=tensorflow/serving:$1-gpu
  docker pull $image
  docker run --rm --runtime=nvidia -p 8500:8500 \
    --mount type=bind,source="$saved_model_path_to_serve",target=/models/mymodel \
    -e MODEL_NAME=mymodel -t $image &
}

run_client() {
  curl -O https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/half_plus_two_client.py
  python half_plus_two_client.py
}

mkdir -p $WORK_DIR
cd $WORK_DIR
mode="$1"
shift

if [[ "$mode" == "server" ]]; then
  run_server "$@"
elif [[ "$mode" == "client" ]]; then
  run_client "$@"
elif [[ "$mode" == "clean" ]]; then
  rm -rf $WORK_DIR
else
  echo "Usage: half_plus_two.sh server|client|clean"
fi
