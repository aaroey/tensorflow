#!/bin/bash
set -ex

# Prerequisites: install tensorflow with version >=1.12
WORK_DIR=/tmp/trt_saved_model_resnet50

run_server() {
  # Should be one of:
  # - resnet_v2_fp32_savedmodel_NCHW
  # - resnet_v2_fp32_savedmodel_NCHW_jpg
  # - resnet_v2_fp32_savedmodel_NHWC
  # - resnet_v2_fp32_savedmodel_NHWC_jpg
  local model=${MODEL:-resnet_v2_fp32_savedmodel_NCHW}
  local saved_model_url=http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/$model.tar.gz
  local saved_model_path=$WORK_DIR/$model
  mkdir -p $saved_model_path

  if ! [[ -f $model.tar.gz ]]; then
    curl -O $saved_model_url
    tar zxf $model.tar.gz
  fi

  local use_trt=${USE_TRT:-true}
  local saved_model_path_to_serve="$saved_model_path"

  if [[ "${use_trt}" == 'true' ]]; then
    local precision_mode=${PRECISION_MODE:-FP32}
    local batch_size=${BATCH_SIZE:-1}
    local is_dynamic_op=${IS_DYNAMIC_OP:-False}
    local trt_saved_model_path=${saved_model_path}_trt_precisionmode${precision_mode}_batchsize${batch_size}_isdynamicop${is_dynamic_op}
    saved_model_path_to_serve="$trt_saved_model_path"
    if ! [[ -f $trt_saved_model_path/1/saved_model.pb ]]; then
      rm -rf $trt_saved_model_path
      local saved_model_path_with_version="$(echo $saved_model_path/*)"

      local tf_tag=tensorflow/tensorflow:1.12.0-gpu
      docker pull $tf_tag
      # TODO(aaroey): uncomment this when 1.13 is out.
      # docker run --runtime=nvidia --rm -v $WORK_DIR:$WORK_DIR -t $tf_tag \
      #   /usr/local/bin/saved_model_cli       \
      #   convert                              \
      #   --dir $saved_model_path_with_version \
      #   --output_dir $trt_saved_model_path/1 \
      #   --tag_set serve                      \
      #   tensorrt                             \
      #   --max_batch_size $batch_size         \
      #   --precision_mode $precision_mode     \
      #   --is_dynamic_op $is_dynamic_op
      docker run --runtime=nvidia --rm -v $WORK_DIR:$WORK_DIR -t $tf_tag \
        python -c "
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=$batch_size,
    precision_mode='$precision_mode',
    is_dynamic_op=$is_dynamic_op,
    input_saved_model_dir='$saved_model_path_with_version',
    output_saved_model_dir='$trt_saved_model_path/1')  # Hard coded version 1
"
    fi
  fi
  echo "----------------------------> model = $model"
  echo "----------------------------> use_trt = $use_trt"
  echo "----------------------------> precision_mode = $precision_mode"
  echo "----------------------------> batch_size = $batch_size"
  echo "----------------------------> is_dynamic_op = $is_dynamic_op"

  # NOTE(aaroey): this should work with latest
  local tfs_tag="${1:-1.12.0}"
  if [[ "$tfs_tag" == 'local' ]]; then
    curl -O https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/tools/docker/Dockerfile.devel-gpu
    curl -O https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/tools/docker/Dockerfile.gpu
    docker build --pull -t my-tfs-devel-gpu \
      -f Dockerfile.devel-gpu \
      .
    docker build -t my-tfs-gpu \
      --build-arg TF_SERVING_BUILD_IMAGE=my-tfs-devel-gpu \
      -f Dockerfile.gpu \
      .
    tfs_tag=my-tfs-gpu
  else
    tfs_tag="tensorflow/serving:${tfs_tag}-gpu"
    docker pull $tfs_tag
  fi

  # NOTE: 8500 is GRPC port, 8501 is REST port, the -p option binds the docker
  # port with host port, if not bind we'll not be able to access that port from
  # host.
  #
  # To debug the model server, add `--cap-add=SYS_PTRACE` to the `docker run`
  # command, and attach to it via `docker exec -i -t <container id> /bin/bash`
  # and then `gdb`.
  docker run --runtime=nvidia --rm -p 8500:8500 -p 8501:8501 \
    -v $saved_model_path_to_serve:/models/mymodel \
    -e MODEL_NAME=mymodel -t $tfs_tag &
}

run_client() {
  curl -O https://raw.githubusercontent.com/aaroey/tensorflow/trt_savedmodel_example/trt_savedmodel_example/resnet50_client.py

  local tfs_tag=tensorflow/serving:1.12.0-devel-gpu
  docker pull $tfs_tag
  docker run --network host --runtime=nvidia --rm -v $WORK_DIR:$WORK_DIR -t $tfs_tag \
    bash -c "
pip install Pillow;
python $WORK_DIR/resnet50_client.py \
  --batch_size=${BATCH_SIZE:-1}     \
  --preprocess_image=${PREPROCESS_IMAGE:-True}
"
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
