#!/bin/bash
set -x

TFTRT_MASKRCNN_TAG=lam8da/aaroey-tensorflow:tf-trt-maskrcnn-test-with-repro

build() {
  mkdir /tmp/ddd
  pushd /tmp/ddd
  docker build --pull -t lam8da/aaroey-tensorflow:tf-trt-maskrcnn-test-with-repro -f ~/Workspace/aaroey/mytfpy3/instructions/devel-gpu.Dockerfile ./
  popd -
}
convert() {
  rm -rf /tmp/maskrcnn-trt

  docker run --runtime=nvidia --rm -v /tmp:/tmp -it $TFTRT_MASKRCNN_TAG \
    /usr/local/bin/saved_model_cli convert \
    --dir /tmp/maskrcnn/1551814896 \
    --output_dir /tmp/maskrcnn-trt \
    --tag_set serve \
    tensorrt \
    --minimum_segment_size 3 \
    --max_workspace_size_bytes 1073741824 \
    --precision_mode FP16 \
    --is_dynamic_op True
}
run() {
  docker run --runtime=nvidia --rm -v /tmp:/tmp \
    -e TF_CPP_VMODULE=convert_nodes=1,segment=1,trt_logger=1 \
    -e TF_CPP_MIN_VLOG_LEVEL=0 \
    -e num_threads=4 \
    -e num_requests=200 \
    -e model_dir=/tmp/maskrcnn-trt \
    -e input_file=/tmp/cat.1472x896.jpg \
    -e resize_to_width=1472 \
    -e resize_to_height=896 \
    -it $TFTRT_MASKRCNN_TAG \
    /tensorflow/bazel-bin/maskrcnn/profile_maskrcnn_cc
}

build
convert
run
