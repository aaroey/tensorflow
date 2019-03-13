#!/bin/bash
set -x

export LD_LIBRARY_PATH=$HOME/Downloads/TensorRT-5.1.2.1-cuda-10.0/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64

# vmodule_config='convert_nodes=1,trt_engine_op=1,segment=1,trt_logger=1'
vmodule_config='convert_nodes=1,segment=1,trt_logger=1'
trt_saved_model_dir=/tmp/maskrcnn-trt

if [[ "$1" == 'py' ]]; then
  rm -rf $trt_saved_model_dir

  ../lambda-install.sh bazelbuild instructions:profile_maskrcnn_py && \
    TF_CPP_VMODULE=$vmodule_config                                    \
    bazel-bin/instructions/profile_maskrcnn_py                        \

else
  gdb=$1

  ../lambda-install.sh bazelbuild instructions:profile_maskrcnn_cc && \
    TF_CPP_VMODULE=$vmodule_config                                    \
    TF_CPP_MIN_VLOG_LEVEL=0                                           \
                                                                      \
    num_threads=1                                                     \
    num_requests=200                                                  \
    model_dir=$trt_saved_model_dir                                    \
    xprof_output_path=$trt_saved_model_dir/gpuprof                    \
    xprof_num_requests=3                                              \
                                                                      \
    input_file=instructions/cat.1472x896.jpg                          \
    output_file_for_actual_input=/tmp/dbg.jpg                         \
    resize_to_width=1472                                              \
    resize_to_height=896                                              \
                                                                      \
    minimum_segment_size=10                                           \
    max_batch_size=1                                                  \
    max_workspace_size_bytes=1073741824                               \
    precision_mode=FP16                                               \
    ${gdb} bazel-bin/instructions/profile_maskrcnn_cc                 \

fi
