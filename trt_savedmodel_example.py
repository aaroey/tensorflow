from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib import tensorrt  # Import the optimizer
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode):
  input_data = tf.reshape(features['x'], (-1, 1, 28, 28))
  x = input_data

  data_format = 'channels_first'

  x = tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=(3, 3),
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)
  x = tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=(3, 3),
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)
  x = tf.layers.max_pooling2d(
      inputs=x, pool_size=(2, 2), strides=2, data_format=data_format)

  x = tf.layers.conv2d(
      inputs=x,
      filters=64,
      kernel_size=(3, 3),
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)
  x = tf.layers.conv2d(
      inputs=x,
      filters=64,
      kernel_size=(3, 3),
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)
  x = tf.layers.max_pooling2d(
      inputs=x, pool_size=(2, 2), strides=2, data_format=data_format)

  x = tf.reshape(x, [-1, 64 * 7 * 7])
  x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
  x = tf.layers.dropout(
      inputs=x, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))

  logits = tf.layers.dense(inputs=x, units=10)

  if labels is not None:
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  elif mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  else:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predictions': tf.estimator.export.PredictOutput(predictions)
        })


def train():
  mnist = tf.contrib.learn.datasets.load_dataset('mnist')
  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir='./log')

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data},
      y=train_labels,
      batch_size=128,
      num_epochs=None,
      shuffle=True)

  mnist_classifier.train(input_fn=train_input_fn, steps=200)
  saved_model_dir = mnist_classifier.export_savedmodel(
      './log/export',
      tf.estimator.export.build_raw_serving_input_receiver_fn({
          'x': tf.placeholder(shape=[None, 784], dtype=tf.float32)
      }))
  print('=========== exported savedmodel to: %s' % saved_model_dir)
  return saved_model_dir


def main(argv):
  saved_model_dir = train()
  batch_size = 16

  # Configure the optimizer options to include TensorRTOptimizer.
  rewriter_cfg = rewriter_config_pb2.RewriterConfig()
  rewriter_cfg.optimizers.extend(['constfold', 'layout'])
  optimizer = rewriter_cfg.custom_optimizers.add()
  optimizer.name = 'TensorRTOptimizer'
  optimizer.parameter_map['minimum_segment_size'].i = 2
  optimizer.parameter_map['max_batch_size'].i = batch_size
  optimizer.parameter_map['is_dynamic_op'].b = False
  optimizer.parameter_map['max_workspace_size_bytes'].i = 1 << 30
  optimizer.parameter_map['precision_mode'].s = 'FP32'
  graph_options = tf.GraphOptions(rewrite_options=rewriter_cfg)
  config = tf.ConfigProto(graph_options=graph_options)

  run_metadata = tf.RunMetadata()
  run_options = tf.RunOptions()
  run_options.output_partition_graphs = True

  # Run the graph with the configured options.
  with tf.Session(config=config) as sess:
    loader.load(sess, [tag_constants.SERVING], saved_model_dir)
    result = sess.run(['softmax_tensor:0'],
                      feed_dict={
                          'Placeholder:0':
                              np.random.random_sample([batch_size, 784]),
                      },
                      options=run_options,
                      run_metadata=run_metadata)
    print(result)

  for gd in run_metadata.partition_graphs:
    device = gd.node[0].device.replace('/', '_').replace(':', '_')
    graph_filename = 'log-graph-%s.pbtxt' % device
    # We can see from the written GPU graph that the trt op is created.
    tf.train.write_graph(gd, '/tmp/lambdatfpy', graph_filename, as_text=True)


if __name__ == '__main__':
  tf.app.run()
