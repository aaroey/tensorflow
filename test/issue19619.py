#!/usr/bin/env python

import argparse
import subprocess
import tensorflow as tf
import numpy as np
import six

from tensorflow.core.protobuf import config_pb2 as cpb2
from tensorflow.core.protobuf import rewriter_config_pb2 as rwpb2

from tensorflow.contrib import tensorrt as trt

NUM_IMAGES = 3

INPUT_TENSORS = [
    u'placeholders/image_0', u'placeholders/image_1', u'placeholders/image_2',
    u'placeholders/num_images', u'placeholders/blend_coeff'
]

OUTPUT_TENSORS = [
    u'detections/center_x', u'detections/center_y', u'detections/width',
    u'detections/height', u'detections/width_3d', u'detections/height_3d',
    u'detections/depth_3d', u'detections/class_id', u'detections/probability',
    u'detections/yaw', u'detections/properties/tl_rotation/class_id',
    u'detections/properties/tl_type/class_id',
    u'detections/properties/tl_road_tl_state/class_id',
    u'detections/properties/tl_bicycle_tl_state/class_id',
    u'detections/properties/tl_pedestrian_tl_state/class_id',
    u'detections/properties/tl_other_tl_state/class_id',
    u'detections/properties/tl_left_section/class_id',
    u'detections/properties/tl_left_section_state/class_id',
    u'detections/properties/tl_right_section/class_id',
    u'detections/properties/tl_right_section_state/class_id',
    u'segmentations/sdc1/0', u'segmentations/sdc1/1', u'segmentations/sdc1/2',
    u'visualizations/sdc1/0', u'visualizations/sdc1/1', u'visualizations/sdc1/2'
]


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--mode', choices=['tf', 'trt'], required=True, help='Evaluator to use.')

  args = parser.parse_args()

  graph_def = tf.GraphDef()
  with tf.gfile.GFile('/tmp/lambdatfpy/trt-ssd-minimal-example/ssd-tensorflow.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
  print('original nodes: %d'%len(graph_def.node))

  converted_graph_def = trt.create_inference_graph(
        input_graph_def=graph_def,
        outputs=INPUT_TENSORS+OUTPUT_TENSORS,
        max_batch_size=3,
        max_workspace_size_bytes=1<<25,
        precision_mode='FP32',
        minimum_segment_size=10,
        is_dynamic_op=False,
        maximum_cached_engines=1)
  print('converted nodes: %d'%len(converted_graph_def.node))
  return

  with tf.Graph().as_default():
    tf.import_graph_def(graph_def, name='')
    run_graph(mode=args.mode)


def run_graph(ntimes=100, mode='trt'):
  image = np.zeros((1024, 768, 3), dtype=np.uint8)
  feed_dict = {tensor_name: image for tensor_name in INPUT_TENSORS[:NUM_IMAGES]}
  feed_dict[u'placeholders/num_images'] = NUM_IMAGES
  feed_dict[u'placeholders/blend_coeff'] = 0.5

  feed_dict = {
      tf.get_default_graph().get_operation_by_name(key).outputs[0]: value
      for key, value in six.iteritems(feed_dict)
  }

  output_tensors = [
      tf.get_default_graph().get_operation_by_name(tensor_name).outputs[0]
      for tensor_name in OUTPUT_TENSORS
  ]
  opt_config = rwpb2.RewriterConfig()
  opt_config.meta_optimizer_iterations = opt_config.ONE
  opt_config.optimizers.extend(['constfold', 'layout'])
  custom_op = opt_config.custom_optimizers.add()
  custom_op.name = 'TensorRTOptimizer'
  custom_op.parameter_map['minimum_segment_size'].i = 10
  custom_op.parameter_map['precision_mode'].s = b'FP32'
  custom_op.parameter_map['max_batch_size'].i = NUM_IMAGES
  custom_op.parameter_map['is_dynamic_op'].b = True
  custom_op.parameter_map['max_workspace_size_bytes'].i = 1 << 25

  graph_options = cpb2.GraphOptions(rewrite_options=opt_config)
  if mode == 'trt':
    config = tf.ConfigProto(graph_options=graph_options)
  else:
    config = tf.ConfigProto()

  with tf.Session(config=config) as session:
    for i in six.moves.xrange(ntimes):
      session.run(output_tensors, feed_dict)
    subprocess.check_call(['nvidia-smi'], close_fds=True)


if __name__ == '__main__':
  main()
