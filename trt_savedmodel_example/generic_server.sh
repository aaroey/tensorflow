#!/bin/bash
set -ex

WORK_DIR=/tmp/trt_saved_model_generic

build_tfs() {
  local tag=$1
  docker build --pull -t ${tag}-devel \
    -f ${DOCKER_FILE_DIR}/Dockerfile.devel-gpu .
  docker build -t $tag --build-arg TF_SERVING_BUILD_IMAGE=${tag}-devel \
    -f ${DOCKER_FILE_DIR}/Dockerfile.gpu .
}

run_server() {
  local url_prefix=http://download.tensorflow.org/models/official/20181001_resnet/savedmodels
  local saved_model_url="${1:-$url_prefix/resnet_v2_fp32_savedmodel_NCHW.tar.gz}"
  local saved_model_name="${saved_model_url##*/}"
  saved_model_name="${saved_model_name%%.tar.gz}"

  local saved_model_path=$WORK_DIR/$saved_model_name
  local trt_saved_model_path=${saved_model_path}_trt
  if [[ "$BATCH_SIZE" ]]; then
    rm -rf $trt_saved_model_path
  fi
  mkdir -p $saved_model_path $trt_saved_model_path

  if ! [[ -f $WORK_DIR/$saved_model_name.tar.gz ]]; then
    curl -O $saved_model_url
    tar zxf $saved_model_name.tar.gz
  fi
  local saved_model_path_to_serve=$saved_model_path

  local tag="${DOCKER_TAG:-latest}"
  local devel_image=tensorflow/serving:$tag-devel-gpu
  local image=tensorflow/serving:$tag-gpu
  if [[ "$tag" == 'local' ]]; then
    image=my-tf-serving-trt
    devel_image=${image}-devel
    build_tfs $image
  else
    docker pull $devel_image
    docker pull $image
  fi

  local run_options="--rm --runtime=nvidia -it --network=host -v $(pwd):$(pwd) -v /tmp:/tmp"
  if [[ "$BATCH_SIZE" ]]; then
    docker run $run_options $devel_image bash -c "
      rm -rf $saved_model_path $trt_saved_model_path
      cd $(pwd)
      pip install tensorflow-gpu

      curl -O $saved_model_url
      tar zxf $model.tar.gz

      python -c '
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=${BATCH_SIZE:-1},
    precision_mode=\"${PRECISION_MODE:-FP32}\",
    is_dynamic_op=${IS_DYNAMIC_OP:-False},
    input_saved_model_dir=\"$(echo $saved_model_path/*)\",
    output_saved_model_dir=\"$trt_saved_model_path/1\")  # Hard coded version 1
'
    "
    saved_model_path_to_serve=$trt_saved_model_path
  fi

  docker run --cap-add=SYS_PTRACE --runtime=nvidia -p 8500:8500 \
    --mount type=bind,source="$saved_model_path_to_serve",target=/models/mymodel \
    -e MODEL_NAME=mymodel -t $tag &
}

mkdir -p $WORK_DIR
cd $WORK_DIR
run_server "$@"
