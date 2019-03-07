#!/bin/bash
set -x


if [[ "$1" == 'py' ]]; then
  rm -rf /tmp/maskrcnn-trt
  ../lambda-install.sh bazelbuild instructions:profile_maskrcnn_py &&     \
    TF_CPP_VMODULE=convert_nodes=1,trt_engine_op=1,segment=1,trt_logger=1 \
    bazel-bin/instructions/profile_maskrcnn_py
else
  gdb=$1
  parent_dir=$HOME/Workspace
  model=$parent_dir/trt_model-tpu-coco-batchnms-normalized-nopreprocess_1551292076_FP16_minsegmentsize3_maxworkspace1G_dynamicop_trtnms
  # model=$parent_dir/trt_model-tpu-coco-batchnms-normalized-nopreprocess_1551292076_FP16_minsegmentsize3_maxworkspace1G_dynamicop

  ../lambda-install.sh bazelbuild instructions:profile_maskrcnn_cc && \
    TF_CPP_VMODULE=convert_nodes=1,trt_engine_op=1,segment=1,trt_logger=1 \
    TF_CPP_MIN_VLOG_LEVEL=0                               \
                                                          \
    num_threads=1                                         \
    num_requests=100                                      \
    model_dir=$model                                      \
    xprof_output_path=$model/gpuprof                      \
    xprof_num_requests=3                                  \
                                                          \
    input_file=$parent_dir/a.jpg                          \
    output_file_for_actual_input=/tmp/dbg.jpg             \
    resize_to_width=1472                                  \
    resize_to_height=896                                  \
                                                          \
    minimum_segment_size=10                               \
    max_batch_size=1                                      \
    max_workspace_size_bytes=1073741824                   \
    precision_mode=FP16                                   \
    ${gdb} bazel-bin/instructions/profile_maskrcnn_cc     \

fi
