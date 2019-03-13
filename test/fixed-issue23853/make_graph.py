#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import six
import tensorflow as tf

from tensorflow.contrib import tensorrt as trt


def main():
  with tf.Graph().as_default(), tf.Session() as session:
    batch_size = 1
    net = tf.placeholder(
        shape=(batch_size, 256, 256, 3), dtype=tf.float32, name='input')

    for i in six.moves.xrange(10):
      in_channels = net.get_shape()[-1].value
      out_channels = 16
      net = tf.nn.conv2d(
          net,
          filter=np.random.rand(3, 3, in_channels, out_channels),
          strides=(1, 1, 1, 1),
          padding='SAME')
      net = tf.nn.relu(net)

    net = tf.identity(net, name='output')
    in_gdef = session.graph.as_graph_def()

  output_node_names = [_get_node_name(net)]

  out_gdef = trt.create_inference_graph(
      input_graph_def=in_gdef,
      outputs=output_node_names,
      max_batch_size=batch_size,
      max_workspace_size_bytes=10**9,
      precision_mode='FP32',
      minimum_segment_size=10)

  serialized_graph_def = out_gdef.SerializeToString()
  # print(serialized_graph_def)
  # assert 'my_trt_op' in serialized_graph_def

  with tf.gfile.GFile('/tmp/lambdatfpy/issue23853.in_gdef.pbtxt', 'w') as f:
    f.write(str(in_gdef))
  with tf.gfile.GFile('/tmp/lambdatfpy/issue23853.out_gdef.pbtxt', 'w') as f:
    f.write(str(out_gdef))
  with tf.gfile.GFile('graph.pb', 'wb') as f:
    f.write(serialized_graph_def)


def _get_node_name(tensor):
  assert tensor.name.endswith(':0')
  return tensor.name[:-len(':0')]


if __name__ == '__main__':
  main()
