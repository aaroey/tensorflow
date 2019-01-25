from PIL import Image
from io import BytesIO
import datetime
import numpy as np
import requests
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants

SAVED_MODEL_DIR = ''
SAVED_MODEL_DIR_TRT = SAVED_MODEL_DIR + '_trt'
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


trt.create_inference_graph(
    None,
    None,
    input_saved_model_dir=SAVED_MODEL_DIR,
    output_saved_model_dir=SAVED_MODEL_DIR_TRT,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 30,
    precision_mode='FP16',
    minimum_segment_size=50,
    is_dynamic_op=True)

time_original = benchmark_model(SAVED_MODEL_DIR)
time_trt = benchmark_model(SAVED_MODEL_DIR_TRT)

for key, t in zip(['original', 'trt'], [time_original, time_trt]):
  print('Time for {} predictions for {} model is {}'.format(NUM_RUNS, key, t))
