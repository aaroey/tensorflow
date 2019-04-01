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
ARG UBUNTU_VERSION=16.04
ARG CUDA_MAJOR_VERSION=9
ARG CUDA_MINOR_VERSION=0

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}-base-ubuntu${UBUNTU_VERSION} as base

# We need to re-declare the ARGs after a FROM instruction, see
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG UBUNTU_VERSION
ARG CUDA_MAJOR_VERSION
ARG CUDA_MINOR_VERSION

# See https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/ for the available cudnn and tensorrt versions.
ARG CUDNN_MAJOR_VERSION=7
ARG CUDNN_VERSION_SUFFIX=5.0.56

# CUDA dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cuda-command-line-tools-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
        cuda-cublas-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
        cuda-cudart-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
        cuda-cufft-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
        cuda-curand-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
        cuda-cusolver-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
        cuda-cusparse-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
        libcudnn7=${CUDNN_MAJOR_VERSION}.${CUDNN_VERSION_SUFFIX}-1+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} \
        libcudnn7-dev=${CUDNN_MAJOR_VERSION}.${CUDNN_VERSION_SUFFIX}-1+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip zip zlib1g-dev wget curl git tmux vim iputils-ping \
        libcupti-dev libtool automake \
        && \
    find /usr/local/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v${CUDNN_MAJOR_VERSION}.a

# TensorRT will be installed in /usr/lib/x86_64-linux-gnu
ARG TENSORRT_VERSION=5.1.2
ENV NVIDIA_ML_REPO=https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64
RUN mkdir /nvinfer && \
    wget -O /nvinfer/libnvinfer.deb ${NVIDIA_ML_REPO}/libnvinfer5_${TENSORRT_VERSION}-1+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}_amd64.deb && \
    wget -O /nvinfer/libnvinfer-dev.deb ${NVIDIA_ML_REPO}/libnvinfer-dev_${TENSORRT_VERSION}-1+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}_amd64.deb && \
    dpkg -i /nvinfer/libnvinfer.deb /nvinfer/libnvinfer-dev.deb && \
    rm -rf /nvinfer

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA=1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=7.0
ENV TF_CUDA_VERSION=${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
ENV TF_CUDNN_VERSION=${CUDNN_MAJOR_VERSION}

ARG USE_PYTHON_3_NOT_2=1
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip \
    ${PYTHON}-dev \
    ${PYTHON}-numpy \
    ${PYTHON}-wheel \
    ${PYTHON}-virtualenv \
    ${PYTHON}-tk

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN apt-get update && apt-get install -y \
    openjdk-8-jdk \
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
ARG BAZEL_VERSION=0.22.0
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
RUN git clone https://github.com/aaroey/tensorflow.git .
RUN git checkout master

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

ENV REPRO_CODE=https://raw.githubusercontent.com/aaroey/tensorflow/maskrcnn_trt/instructions
RUN mkdir maskrcnn
ADD ${REPRO_CODE}/profile_maskrcnn.cc maskrcnn/
ADD ${REPRO_CODE}/BUILD maskrcnn/BUILD
RUN bazel build -c opt --copt=-mavx --config=cuda \
        --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        maskrcnn:profile_maskrcnn_cc
