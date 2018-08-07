import numpy as np
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt


def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


graph_def = load_graph("/tmp/lambdatfpy/simplegdef.pb")
with tf.Graph().as_default():
  tf.import_graph_def(graph_def, name="")
  with tf.Session() as sess:
    print(sess.run("input:0", feed_dict={"input:0": np.zeros((1, 24, 24, 2))}))

trt_graph = trt.create_inference_graph(
    input_graph_def=graph_def,
    outputs=["output"],
    max_batch_size=5,
    max_workspace_size_bytes=5 << 25,
    precision_mode="FP16",
    minimum_segment_size=2)

print("-" * 100 + "> converted graph nodes:")
for n in trt_graph.node:
  print(n.name)
print("-" * 100)

with tf.Graph().as_default():
  tf.import_graph_def(trt_graph, name="")
  with tf.Session() as sess:
    print(sess.run(
        "my_trt_op_0:0", feed_dict={"input:0": np.zeros((1, 24, 24, 2))}))
