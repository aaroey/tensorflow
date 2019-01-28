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

from PIL import Image
from io import BytesIO
import datetime
import numpy as np
import requests
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants

SAVED_MODEL_DIR = ''
SAVED_MODEL_DIR_TRT = SAVED_MODEL_DIR + '_trt'

try:
  import shutil
  shutil.rmtree(SAVED_MODEL_DIR_TRT)
except Exception as e:
  tf.logging.info(e)

IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
NUM_RUNS = 100

get_image_response = requests.get(IMAGE_URL)
outputs = [
    'map_2/TensorArrayStack/TensorArrayGatherV3:0',
    'Sigmoid:0',
    'stack_3:0',
]


def benchmark_model(model_path):
  g = tf.Graph()
  with g.as_default():
    with tf.Session() as sess:
      loader.load(sess, [tag_constants.SERVING], model_path)
      result = sess.run(
          outputs, feed_dict={'Placeholder:0': [get_image_response.content]})
      dt0 = datetime.datetime.now()
      for i in range(NUM_RUNS):
        result = sess.run(
            outputs, feed_dict={'Placeholder:0': [get_image_response.content]})
      dt1 = datetime.datetime.now()
      return dt1 - dt0


session_config = tf.ConfigProto()
rewriter_config = session_config.graph_options.rewrite_options
rewriter_config.optimizers.extend([
    'constfold', 'layout', 'constfold', 'arithmetic', 'constfold', 'arithmetic',
    'constfold'
])
rewriter_config.meta_optimizer_iterations = (
    rewriter_config_pb2.RewriterConfig.ONE)

trt.create_inference_graph(
    None,
    None,
    input_saved_model_dir=SAVED_MODEL_DIR,
    output_saved_model_dir=SAVED_MODEL_DIR_TRT,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 30,
    precision_mode='FP16',
    minimum_segment_size=50,
    is_dynamic_op=True,
    session_config=session_config)

time_original = benchmark_model(SAVED_MODEL_DIR)
time_trt = benchmark_model(SAVED_MODEL_DIR_TRT)

for key, t in zip(['original', 'trt'], [time_original, time_trt]):
  print('Time for {} predictions for {} model is {}'.format(NUM_RUNS, key, t))

# Result with a Titan-V:
# Time for 100 predictions for original model is 0:00:13.118136
# Time for 100 predictions for trt model is 0:00:10.304332
