# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================

# Instructions:
# 1. download this script
# 2. copy the trained MaskRCNN SavedModel to /tmp/maskrcnn
# 3. run this script
# 4. (optional) run this script by passing a 'rebuild' argument if we want to
#    rebuild the docker image instead of pulling it from docker hub.
set -x

GITHUB_URL_PREFIX=https://raw.githubusercontent.com/aaroey/tensorflow/maskrcnn_trt/instructions
TFTRT_MASKRCNN_TAG=lam8da/aaroey-tensorflow:tf-trt-maskrcnn-test

WORKING_DIR=/tmp/maskrcnntest
SAVED_MODEL_DIR=/tmp/maskrcnn
TRT_SAVED_MODEL_DIR=/tmp/maskrcnn-trt
mkdir -p $WORKING_DIR
cd $WORKING_DIR

build() {
  curl -O $GITHUB_URL_PREFIX/devel-gpu.Dockerfile
  docker build --pull -t $TFTRT_MASKRCNN_TAG -f devel-gpu.Dockerfile ./
}
convert() {
  curl -O $GITHUB_URL_PREFIX/convert_maskrcnn.py
  docker run --runtime=nvidia --rm -v /tmp:/tmp -it $TFTRT_MASKRCNN_TAG \
    bash -c "rm -rf $TRT_SAVED_MODEL_DIR;
    python $WORKING_DIR/convert_maskrcnn.py $SAVED_MODEL_DIR $TRT_SAVED_MODEL_DIR"
}
run() {
  curl -O $GITHUB_URL_PREFIX/cat.1472x896.jpg
  docker run --runtime=nvidia --rm -v /tmp:/tmp \
    -e TF_CPP_VMODULE=convert_nodes=1,segment=1,trt_logger=1 \
    -e TF_CPP_MIN_VLOG_LEVEL=0 \
    -e num_threads=4 \
    -e num_requests=200 \
    -e model_dir=$TRT_SAVED_MODEL_DIR \
    -e input_file=$WORKING_DIR/cat.1472x896.jpg \
    -e resize_to_width=1472 \
    -e resize_to_height=896 \
    -it $TFTRT_MASKRCNN_TAG \
    /tensorflow/bazel-bin/maskrcnn/profile_maskrcnn_cc
}

if [[ "$1" == 'rebuild' ]]; then
  build
fi
