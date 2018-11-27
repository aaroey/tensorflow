#!/bin/bash
set -x
# sudo rm -rf /tmp/resnet_trt

mkdir /tmp/resnet
curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz | tar --strip-components=2 -C /tmp/resnet -xvz


ls /tmp/resnet


if [[ "$1" ]]; then
  docker pull tensorflow/serving:latest-devel-gpu
  docker run --rm --runtime=nvidia -it --network=host -v /tmp:/tmp \
    tensorflow/serving:latest-devel-gpu bash -c "
  pip install tensorflow-gpu

  python -c '
import tensorflow.contrib.tensorrt as trt
trt.create_inference_graph(
    None,
    None,
    max_batch_size=1,
    precision_mode=\"FP32\",
    input_saved_model_dir=\"$(echo /tmp/resnet/*)\",
    output_saved_model_dir=\"/tmp/resnet_trt/1\")  # Hard coded version 1
'
  "
fi


docker pull tensorflow/serving:latest-gpu
docker run --rm --runtime=nvidia -p 8501:8501 --name tfserving_resnet \
  --mount type=bind,source=/tmp/resnet_trt,target=/models/resnet \
  -e MODEL_NAME=resnet -t tensorflow/serving:latest-gpu &
