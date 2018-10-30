#!/bin/bash
#!/bin/bash
set -ex

# Prerequisites: install tensorflow with version >=1.12
WORK_DIR=/tmp/trt_saved_model_half_plus_two

run_server() {
  local saved_model_path=$WORK_DIR/saved_model
  rm -rf $saved_model_path*

  curl -O \
    https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/saved_model/saved_model_half_plus_two.py

  python saved_model_half_plus_two.py            \
    --device=gpu                                 \
    --output_dir=$saved_model_path               \
    --output_dir_pbtxt=${saved_model_path}_pbtxt \
    --output_dir_main_op=${saved_model_path}_main_op

  local trt_saved_model_path=$WORK_DIR/saved_model_trt
  if ! [[ -f $trt_saved_model_path/saved_model.pb ]]; then
    python <<< "
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=1,
    input_saved_model_dir='$saved_model_path',
    output_saved_model_dir='$trt_saved_model_path/1')  # Hard coded version 1
"
  fi

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
  curl -O https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/serve_trt_saved_model_half_plus_two_client.py
  python serve_trt_saved_model_half_plus_two_client.py
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
  echo "Usage: serve_trt_saved_model_half_plus_two.sh server|client|clean"
fi
