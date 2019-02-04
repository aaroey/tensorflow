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

# To reproduce the TF-TRT performance on MaskRCNN model:
#
# $ mkdir /tmp/maskrcnn
# $ cd /tmp/maskrcnn
# $ curl -O https://raw.githubusercontent.com/aaroey/tensorflow/maskrcnn_trt/instructions/maskrcnn_new.py
#
# Then open /tmp/maskrcnn/maskrcnn_new.py and set the SAVED_MODEL_DIR (line 26) to
# the maskrcnn SavedModel path. Then run:
#
# $ docker pull tensorflow/tensorflow:nightly-gpu
# $ docker run --runtime=nvidia --rm -v /tmp:/tmp -it tensorflow/tensorflow:nightly-gpu bash -c "pip install Pillow requests; python /tmp/maskrcnn/maskrcnn_new.py"
#
# ==============================================================================
# Below are old instructions to use customized TF repo (outdated):
#
# $ mkdir /tmp/maskrcnn
# $ cd /tmp/maskrcnn
# $ curl -O https://raw.githubusercontent.com/aaroey/tensorflow/maskrcnn_trt/instructions/maskrcnn.py
# $ export TFTRT_MASKRCNN_TAG=lam8da/aaroey-tensorflow:tf-trt-maskrcnn-test
# $ docker pull $TFTRT_MASKRCNN_TAG
#
# Then open /tmp/maskrcnn/maskrcnn.py and set the SAVED_MODEL_DIR (line 26) to
# the maskrcnn SavedModel path. Then run:
#
# $ docker run --runtime=nvidia --rm -v /tmp:/tmp -it $TFTRT_MASKRCNN_TAG python /tmp/maskrcnn/maskrcnn.py
#
# (For debugging only) Notes: instead of 'docker pull', we can also build
# the docker image using this Dockerfile. The result will be similar, but the
# pulled version is smaller since it cleans up bazel cache after building TF. To
# build, run:
# $ curl -O https://raw.githubusercontent.com/aaroey/tensorflow/maskrcnn_trt/instructions/devel-gpu.Dockerfile
# $ docker build --pull -t $TFTRT_MASKRCNN_TAG -f /tmp/maskrcnn/devel-gpu.Dockerfile ./  # This will take a while

ARG UBUNTU_VERSION=16.04

FROM nvidia/cuda:10.0-base-ubuntu${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cudart-dev-10-0 \
        cuda-cufft-dev-10-0 \
        cuda-curand-dev-10-0 \
        cuda-cusolver-dev-10-0 \
        cuda-cusparse-dev-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0 \
        libcudnn7-dev=7.4.1.5-1+cuda10.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        git \
        && \
    find /usr/local/cuda-10.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0 \
        && rm -rf /var/lib/apt/lists/*

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=6.0,6.1,7.0
ENV TF_CUDA_VERSION=10.0
ENV TF_CUDNN_VERSION=7

ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

RUN ${PIP} --no-cache-dir install \
    Pillow \
    requests \
    h5py \
    keras_applications \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
    enum34

# Install bazel
ARG BAZEL_VERSION=0.19.2
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh

# COPY bashrc /etc/bash.bashrc
# RUN chmod a+rwx /etc/bash.bashrc

# Check out and build TensorFlow source code
RUN mkdir /tensorflow
WORKDIR /tensorflow
RUN git clone --branch=maskrcnn_trt --depth=1 https://github.com/aaroey/tensorflow.git .
RUN git checkout maskrcnn_trt

# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
#     LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
#   rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
RUN tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --config=cuda \
        --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /tmp/pip
# Clean up pip wheel (but not Bazel cache) when done.
# rm -rf /root/.cache
