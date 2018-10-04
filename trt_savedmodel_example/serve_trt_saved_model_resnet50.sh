#!/bin/bash
set -ex

# Prerequisites: install tensorflow with version >=1.12
work_dir=/tmp/trt_saved_model
mkdir -p $work_dir
cd $work_dir

if [[ "$1" == "server" ]]; then
  saved_model_name=20180601_resnet_v2_imagenet_savedmodel
  saved_model_url=http://download.tensorflow.org/models/official/$saved_model_name.tar.gz
  saved_model_path=$work_dir/$saved_model_name
  trt_saved_model_path=${saved_model_path}_trt

  if ! [[ -d $saved_model_path ]]; then
    curl -O $saved_model_url
    tar zxf $saved_model_name.tar.gz
  fi

  if ! [[ -f $trt_saved_model_path/1/saved_model.pb ]]; then
    saved_model_path_with_version="$(echo $saved_model_path/*)"
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

  # NOTE: 8500 is GRPC port, 8501 is REST port, the -p option binds the docker
  # port with host port, if not bind we'll not be able to access that port from
  # host.
  docker run --runtime=nvidia -p 8500:8500 \
    --mount type=bind,source="$trt_saved_model_path",target=/models/mymodel \
    -e MODEL_NAME=mymodel -t \
    tensorflow/serving:nightly-gpu &
    # tensorflow/serving:latest-gpu &
elif [[ "$1" == "client" ]]; then
  curl -O https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/serve_trt_saved_model_resnet50_client.py
  python serve_trt_saved_model_resnet50_client.py \
    --num_tests=10 \
    --server=127.0.0.1:8500
else
  echo 'Usage: run_oss.sh server|client'
fi
