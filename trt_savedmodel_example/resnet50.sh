#!/bin/bash
set -ex

# Prerequisites: install tensorflow with version >=1.12
WORK_DIR=/tmp/trt_saved_model_resnet50

run_server() {
  local saved_model_name=resnet_v2_fp32_savedmodel_NCHW
  local saved_model_url=http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz
  local saved_model_path=$WORK_DIR/$saved_model_name
  local trt_saved_model_path=${saved_model_path}_trt
  mkdir -p $saved_model_path $trt_saved_model_path
  rm -rf $saved_model_path $trt_saved_model_path

  curl -O $saved_model_url
  tar zxf $saved_model_name.tar.gz

  local use_trt=${USE_TRT:-'true'}
  local batch_size=${BATCH_SIZE:-1}
  local saved_model_path_to_serve="$saved_model_path"
  if [[ "${use_trt}" == 'true' ]]; then
    saved_model_path_to_serve="$trt_saved_model_path"
    local saved_model_path_with_version="$(echo $saved_model_path/*)"
    python <<< "
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=$batch_size,
    input_saved_model_dir='$saved_model_path_with_version',
    output_saved_model_dir='$trt_saved_model_path/1')  # Hard coded version 1
"
  fi
  echo "use_trt = $use_trt"
  echo "batch_size = $batch_size"

  local tag=''
  if [[ "$1" == 'local' ]]; then
    curl -O https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/tools/docker/Dockerfile.devel-gpu
    curl -O https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/tools/docker/Dockerfile.gpu
    docker build --pull -t my-tf-serving-trt-tf-head-env \
      -f Dockerfile.devel-gpu \
      .
    docker build -t my-tf-serving-trt-devel \
      --build-arg TF_SERVING_BUILD_IMAGE=my-tf-serving-trt-tf-head-env \
      -f Dockerfile.gpu \
      .
    tag=my-tf-serving-trt-devel
  elif [[ "$1" == 'nightly' ]]; then
    tag=tensorflow/serving:nightly-gpu
    docker pull $tag
  else
    tag=tensorflow/serving:latest-gpu
    docker pull $tag
  fi

  # NOTE: 8500 is GRPC port, 8501 is REST port, the -p option binds the docker
  # port with host port, if not bind we'll not be able to access that port from
  # host.
  #
  # To debug the model server, add `--cap-add=SYS_PTRACE` to the `docker run`
  # command, and attach to it via `docker exec -i -t <container id> /bin/bash`
  # and then `gdb`.
  docker run --cap-add=SYS_PTRACE --runtime=nvidia -p 8500:8500 \
    --mount type=bind,source="$saved_model_path_to_serve",target=/models/mymodel \
    -e MODEL_NAME=mymodel -t $tag &
}

run_client() {
  curl -O https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/resnet50_client.py
  python resnet50_client.py --batch_size=${BATCH_SIZE:-1}
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
  echo "Usage: resnet50.sh server|client|clean"
fi
