import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import importer

in_file_name = "frozen_graph_static_shape.pb"
in_gdef = tf.GraphDef()
with open(in_file_name, "r") as f:
  in_gdef.ParseFromString(f.read())

input_name = "placeholder"
output_name = "mul"

trt_gdef = trt.create_inference_graph(
    in_gdef,
    outputs=[output_name],
    max_batch_size=1,
    max_workspace_size_bytes=1 << 30,
    minimum_segment_size=3,
    precision_mode="FP32")

g = tf.Graph()
with g.as_default():
  importer.import_graph_def(trt_gdef, name="")
  input_dim = g.get_tensor_by_name(input_name + ":0").shape
  input_data = np.random.random_sample(input_dim)
  tf.train.write_graph(
      g.as_graph_def(add_shapes=True), "/tmp", "converted.pbtxt")
  with tf.Session(
      config=tf.ConfigProto(
          gpu_options=tf.GPUOptions(
              per_process_gpu_memory_fraction=0.7))) as sess:
    val = sess.run(output_name + ":0", {input_name + ":0": input_data})
    print(val)

# Importing the converted graph, also fail:
# in_file_name = "/tmp/converted.pbtxt"
# in_gdef = tf.GraphDef()
# with open(in_file_name, "r") as f:
#   from google.protobuf import text_format
#   text_format.Merge(f.read(), in_gdef)
#
# g = tf.Graph()
# with g.as_default():
#   importer.import_graph_def(in_gdef, name="")
#   input_dim = g.get_tensor_by_name(input_name + ":0").shape
#   input_data = np.random.random_sample(input_dim)
#   with tf.Session(
#       config=tf.ConfigProto(
#           gpu_options=tf.GPUOptions(
#               per_process_gpu_memory_fraction=0.7))) as sess:
#     val = sess.run(output_name + ":0", {input_name + ":0": input_data})
#     print(val)
