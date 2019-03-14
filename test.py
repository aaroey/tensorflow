import os
os.chdir('/tmp')

import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import loader
from tensorflow.python.compiler.tensorrt import trt_convert

smdir = '/tmp/simple_saved_model_trt'

with tf.Graph().as_default():
  with tf.Session(
      config=tf.ConfigProto(gpu_options=tf.GPUOptions(
          allow_growth=True))) as sess:
    loader.load(sess, ['serve'], smdir)
    sess.run(
        'output:0',
        feed_dict={'input:0': np.random.random_sample([1, 24, 24, 2])})
