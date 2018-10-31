## Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
r"""Exports an example linear regression inference graph.

Exports a TensorFlow graph to `/tmp/saved_model/half_plus_two/` based on the
`SavedModel` format.

This graph calculates,

\\(
  y = a*x + b
\\)

where `a` and `b` are variables with `a=0.5` and `b=2`.

Output from this program is typically used to exercise SavedModel load and
execution code.

To create a model:
  python saved_model_half_plus_two.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

FLAGS = None
MAX_BATCH_SIZE = 1


def _generate_saved_model_for_half_plus_two(export_dir, use_main_op=False):
  """Generates SavedModel for half plus two.

  Args:
    export_dir: The directory to which the SavedModel should be written.
    use_main_op: Whether to supply a main op during SavedModel build time.
  """
  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
  with tf.Session(
      graph=tf.Graph(),
      config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.device("/gpu:0"):
      # Set up the model parameters as variables to exercise variable loading
      # functionality upon restore.
      a = tf.Variable(0.5, name="a")
      b = tf.Variable(2.0, name="b")

      # Create a placeholder for input parameter x.
      x = tf.placeholder(tf.float32, shape=(MAX_BATCH_SIZE, 1), name="x")

      # Calculate y=a*x+b. Use tf.identity() to assign name
      y = tf.add(tf.multiply(a, x), b)
      y = tf.identity(y, name="y")

    # Set up the signature for Predict with input and output tensor
    # specification.
    predict_signature_inputs = {"x": tf.saved_model.utils.build_tensor_info(x)}
    predict_signature_outputs = {"y": tf.saved_model.utils.build_tensor_info(y)}
    predict_signature_def = (
        tf.saved_model.signature_def_utils.build_signature_def(
            predict_signature_inputs, predict_signature_outputs,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    signature_def_map = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            predict_signature_def
    }

    # Initialize all variables and then save the SavedModel.
    sess.run(tf.global_variables_initializer())
    if use_main_op:
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map=signature_def_map,
          assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
          main_op=tf.saved_model.main_op.main_op())
    else:
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map=signature_def_map,
          assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
  builder.save(as_text=True)


def main(_):
  _generate_saved_model_for_half_plus_two(
      FLAGS.output_dir, use_main_op=FLAGS.use_main_op)
  print("SavedModel generated for GPUs at: %s" % FLAGS.output_dir)

  trt.create_inference_graph(
      None,
      None,
      max_batch_size=MAX_BATCH_SIZE,
      input_saved_model_dir=FLAGS.output_dir,
      output_saved_model_dir=FLAGS.output_dir_trt)
  print("TRT-converted SavedModel generated at: %s" % FLAGS.output_dir_trt)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_dir",
      type=str,
      default="/tmp/saved_model_half_plus_two/1",
      help="Directory where to output SavedModel.")
  parser.add_argument(
      "--output_dir_trt",
      type=str,
      default="/tmp/saved_model_half_plus_two_trt/1",
      help="Directory where to output TRT-converted SavedModel.")
  parser.add_argument(
      "--use_main_op",
      type=bool,
      default=False,
      help="Whether to supply a main op during SavedModel build time.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
