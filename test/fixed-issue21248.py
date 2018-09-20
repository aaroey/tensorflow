import numpy as np
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer


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
  inp = tf.placeholder(tf.float32, shape=(1, 28, 28, 3), name="input_image")
  deconv1 = tf.layers.conv2d_transpose(
      inp, filters=8, kernel_size=(3, 3), strides=(2, 2))
  output = tf.layers.conv2d(
      deconv1, filters=8, kernel_size=(3, 3), name="output")

  input_nodes = ["input_image"]
  output_nodes = ["output/BiasAdd"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    const_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_nodes)
  tf.train.write_graph(const_graph_def, "/tmp/lambdatfpy",
                       "issue21248.const.pbtxt")

  # Run const folding manually
  graph = tf.Graph()
  with graph.as_default():
    tf.import_graph_def(const_graph_def, name="")
  meta_graph = tf.train.export_meta_graph(
      graph_def=graph.as_graph_def(), graph=graph)
  output_collection = meta_graph_pb2.CollectionDef()
  output_list = output_collection.node_list.value
  to_bytes = lambda inp: inp.encode("utf-8", errors="surrogateescape")
  for i in output_nodes:
    if isinstance(i, tf.Tensor):
      output_list.append(to_bytes(i.name))
    else:
      output_list.append(to_bytes(i))
  meta_graph.collection_def["train_op"].CopyFrom(output_collection)
  rewriter_cfg = rewriter_config_pb2.RewriterConfig()
  rewriter_cfg.optimizers.extend(["constfold", "layout"])
  folded_gdef = tf_optimizer.OptimizeGraph(
      rewriter_cfg, meta_graph, graph_id=b"tf_graph")
  tf.train.write_graph(folded_gdef, "/tmp/lambdatfpy",
                       "issue21248.folded.pbtxt")

  optimized_graph_def = trt.create_inference_graph(
      input_graph_def=folded_gdef,
      outputs=output_nodes,
      max_batch_size=1,
      max_workspace_size_bytes=1 << 25)
  tf.train.write_graph(optimized_graph_def, "/tmp/lambdatfpy",
                       "issue21248.optimized.pbtxt")
  graph, input_tensors, output_tensors = build_graph_from_def(
      optimized_graph_def, input_nodes, output_nodes)
  with tf.Session(graph=graph) as sess:
    sess.run(
        output_tensors[0],
        feed_dict={input_tensors[0]: np.zeros((1, 28, 28, 3))})


if __name__ == "__main__":
  main()
