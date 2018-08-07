import numpy as np
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt


def build_graph_from_def(graph_def, input_nodes, output_nodes):
  """
    build the actual graph from definition
    """
  tf.reset_default_graph()
  graph = tf.Graph()
  with graph.as_default():
    return_tensors = [
        operation_name + ":0" for operation_name in (input_nodes + output_nodes)
    ]
    tensors = tf.import_graph_def(
        graph_def=graph_def, name="", return_elements=return_tensors)
    input_tensor_list = tensors[:len(input_nodes)]
    output_tensor_list = tensors[len(input_nodes):]

  return graph, input_tensor_list, output_tensor_list


def main():
  with tf.variable_scope("Net"):
    inp = tf.placeholder(tf.float32, shape=(1, 28, 28, 3), name="input_image")
    deconv1 = tf.layers.conv2d_transpose(
        inp, filters=8, kernel_size=(3, 3), strides=(2, 2))
    output = tf.layers.conv2d(
        deconv1, filters=8, kernel_size=(3, 3), name="output")
  input_nodes = ["Net/input_image"]
  output_nodes = ["Net/output/BiasAdd"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    const_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_nodes)
  tf.train.write_graph(const_graph_def, "/tmp", "issue21248.const.pbtxt")

  optimized_graph_def = trt.create_inference_graph(
      input_graph_def=const_graph_def,
      outputs=output_nodes,
      max_batch_size=1,
      max_workspace_size_bytes=1 << 25)
  tf.train.write_graph(optimized_graph_def, "/tmp",
                       "issue21248.optimized.pbtxt")
  graph, input_tensors, output_tensors = build_graph_from_def(
      optimized_graph_def, input_nodes, output_nodes)
  with tf.Session(graph=graph) as sess:
    sess.run(
        output_tensors[0],
        feed_dict={input_tensors[0]: np.zeros((1, 28, 28, 3))})


if __name__ == "__main__":
  main()
