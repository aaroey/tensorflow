import numpy as np

import tensorflow as tf
from tensorflow.contrib import tensorrt as trt
from tensorflow.python.framework import importer
from tensorflow.python.ops import gen_nn_ops

INPUT_NAME = 'input'
OUTPUT_NAME = 'output'
INPUT_DIMS = [1, 24, 24, 2]


def simple_graphdef(with_gpu=True, dtype=tf.float32):

  def _build_model(inp):
    conv_filter = tf.constant(
        [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
        name='weights',
        dtype=dtype)
    conv = gen_nn_ops.conv2d(
        input=inp,
        filter=conv_filter,
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='conv')
    bias = tf.constant([4., 1.5, 2., 3., 5., 7.], name='bias', dtype=dtype)
    added = gen_nn_ops.bias_add(conv, bias, name='bias_add')
    relu = gen_nn_ops.relu(added, 'relu')
    identity = tf.identity(relu, 'identity')
    pool = tf.nn.max_pool(
        identity, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='max_pool')
    return pool

  g = tf.Graph()
  with g.as_default():
    inp = tf.placeholder(
        dtype=dtype, shape=[None] + INPUT_DIMS[1:], name=INPUT_NAME)
    if with_gpu:
      with g.device('/GPU:0'):
        pool = _build_model(inp)
    else:
      pool = _build_model(inp)
    tf.squeeze(pool, name=OUTPUT_NAME)
  return g.as_graph_def()


def simple_trt_graphdef(with_gpu=True, dtype=tf.float32):
  gdef = simple_graphdef(with_gpu, dtype)
  trt_gdef = trt.create_inference_graph(
      gdef,
      outputs=[OUTPUT_NAME],
      max_batch_size=INPUT_DIMS[0],
      max_workspace_size_bytes=1 << 30,
      minimum_segment_size=2,
      precision_mode='FP32')
  return trt_gdef


def run_gdef(gdef):
  vgd_config = tf.GPUOptions.Experimental(virtual_devices=[
      tf.GPUOptions.Experimental.VirtualDevices(memory_limit_mb=[2048])
  ])
  gpu_options = tf.GPUOptions(
      allow_growth=False, visible_device_list='0', experimental=vgd_config)
  config = tf.ConfigProto(gpu_options=gpu_options)

  g = tf.Graph()
  with g.as_default():
    importer.import_graph_def(gdef, name='')
    with tf.Session(config=config) as sess:
      return sess.run(
          OUTPUT_NAME + ':0',
          feed_dict={INPUT_NAME + ':0': np.random.uniform(size=INPUT_DIMS)})


def myprint(msg):
  print('-' * 50 + msg)


gdef = simple_graphdef()
result = run_gdef(gdef)

trt_gdef = simple_trt_graphdef()
trt_result = run_gdef(trt_gdef)

myprint('-' * 50)
myprint('TF version: %s' % str(tf.__version__))

myprint('max abs result: %s' % str(np.max(np.abs(result))))
myprint('max abs trt_result: %s' % str(np.max(np.abs(trt_result))))
abs_diff = np.abs((result - trt_result))
myprint('max abs diff: %s' % str(np.max(abs_diff)))
myprint('mean abs diff: %s' % str(np.mean(abs_diff)))

myprint('linked_tensorrt_version: %s' % str(
    trt.trt_convert.get_linked_tensorrt_version()))
myprint('loaded_tensorrt_version: %s' % str(
    trt.trt_convert.get_loaded_tensorrt_version()))
