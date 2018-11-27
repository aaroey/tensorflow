#!/bin/bash
set -ex

# Prerequisites: install tensorflow with version >=1.12
WORK_DIR=/tmp/trt_saved_model_resnet50

run_server() {
  local tag="${1:-latest}"
  local devel_image=tensorflow/serving:$tag-devel-gpu
  local image=tensorflow/serving:$tag-gpu
  local run_options="--rm --runtime=nvidia -it --network=host -v $(pwd):$(pwd) -v /tmp:/tmp"

  # Should be one of:
  # - resnet_v2_fp32_savedmodel_NCHW
  # - resnet_v2_fp32_savedmodel_NCHW_jpg
  # - resnet_v2_fp32_savedmodel_NHWC
  # - resnet_v2_fp32_savedmodel_NHWC_jpg
  local model=${MODEL:-resnet_v2_fp32_savedmodel_NCHW}
  local saved_model_url=http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/$model.tar.gz
  local saved_model_path=$WORK_DIR/$model
  local trt_saved_model_path=$WORK_DIR/${model}_trt

  sudo rm -rf $saved_model_path*
  curl -O $saved_model_url
  tar zxf $model.tar.gz
  local saved_model_path_with_version=$(echo $saved_model_path/*)

  docker pull $devel_image
  docker run $run_options $devel_image bash -c "
    cd $(pwd)
    pip install tensorflow-gpu

    python -c '
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=${BATCH_SIZE:-1},
    precision_mode=\"${PRECISION_MODE:-FP32}\",
    is_dynamic_op=${IS_DYNAMIC_OP:-False},
    input_saved_model_dir=\"${saved_model_path_with_version}\",
    output_saved_model_dir=\"$trt_saved_model_path/1\")  # Hard coded version 1
'
  "

  docker pull $image
  docker run --rm --runtime=nvidia -p 8500:8500 -p 8501:8501 \
    --mount type=bind,source="$trt_saved_model_path",target=/models/mymodel \
    -e MODEL_NAME=mymodel -t $image &
}

run_client() {
  curl -O https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/resnet50_client.py

  python resnet50_client.py \
    --batch_size=${BATCH_SIZE:-1} \
    --preprocess_image=${PREPROCESS_IMAGE:-True}
}

mkdir -p $WORK_DIR
cd $WORK_DIR
mode="$1"
shift

if [[ "$mode" == "server" ]]; then
  run_server "$@"
elif [[ "$mode" == "client" ]]; then
  run_client "$@"
  echo "Usage: resnet50.sh server|client|clean"
fi
