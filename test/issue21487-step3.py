import os

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt


def get_frozen_graph(graph_file):
  """Read Frozen Graph file from disk."""
  with tf.gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

def main():
  frozen_graph_def = get_frozen_graph('log/freeze_graph.pb')

  output_nodes = ['softmax_tensor']
  output_dir = 'tensorrt_dir'

  trt_graph_def = trt.create_inference_graph(
    frozen_graph_def,
    output_nodes,
    max_batch_size=1,
    max_workspace_size_bytes=(2 << 10) << 20,
    precision_mode='FP32')

  tf.reset_default_graph()
  g = tf.Graph()

  with g.as_default():
    tf.import_graph_def(
      graph_def=trt_graph_def,
      return_elements=output_nodes,
      name=''
    )

  with tf.Session(graph=g) as sess:
    builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
    builder.add_meta_graph_and_variables(
      sess,
      [tf.saved_model.tag_constants.SERVING]
    )
    builder.save()
    train_writer = tf.summary.FileWriter(output_dir)
    train_writer.add_graph(sess.graph)


if __name__ == '__main__':
  main()
