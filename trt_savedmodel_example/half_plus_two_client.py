# Search for half_plus_two_client.py and that is where this file was copied from
import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('model_name', 'mymodel',
                           'Name of the model being served')
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'Model server address')
tf.app.flags.DEFINE_float('x_value', 1.0, 'x value in y =0.5 * x + 2')
FLAGS = tf.app.flags.FLAGS


def send_regression_request(stub, model_name, signature_name, input_name,
                            input_val):
  request = regression_pb2.RegressionRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = signature_name
  example = request.input.example_list.examples.add()
  example.features.feature[input_name].float_list.value.append(input_val)
  response = stub.Regress(request)
  print('regression result: ' + str(response.result.regressions[0].value))


def send_classification_request(stub, model_name, signature_name, input_name,
                                input_val):
  request = classification_pb2.ClassificationRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = signature_name
  example = request.input.example_list.examples.add()
  example.features.feature[input_name].float_list.value.append(input_val)
  response = stub.Classify(request)
  classes = response.result.classifications[0].classes
  for c in classes:
    print('classification result: label=' + str(c.label) + ', score=' +
          str(c.score))


def send_predict_request(stub, model_name, signature_name, input_name,
                         input_val):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = signature_name
  example = request.inputs[input_name].CopyFrom(
      tf.contrib.util.make_tensor_proto([input_val]))
  response = stub.Predict(request)
  for key in response.outputs:
    print('prediction result: ' + key + ': ' + str(response.outputs[key]))


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # It seems 'regress_x2_to_y3' is broken, since Regress method take an Example
  # but the signature takes a float.
  # send_regression_request(stub, FLAGS.model_name, 'regress_x2_to_y3', 'x2',
  #                         FLAGS.x_value)
  send_regression_request(stub, FLAGS.model_name, 'regress_x_to_y', 'x',
                          FLAGS.x_value)
  send_regression_request(stub, FLAGS.model_name, 'regress_x_to_y2', 'x',
                          FLAGS.x_value)
  send_classification_request(stub, FLAGS.model_name, 'classify_x_to_y', 'x',
                              FLAGS.x_value)
  # TODO(aaroey): there is a problem: TRT fuses two "mostly-disjoint" subgraph
  # (the joint part only contains Const nodes) into same engine, causing that
  # any output of the engine depends on inputs of both subgraphs, while
  # theoretically only input for the same subgraph is required.
  #
  # As a result, sending request to 'serving_default' below with TRT-converted
  # model will fail, but will work fine with original model.
  send_predict_request(stub, FLAGS.model_name, 'serving_default', 'x',
                       FLAGS.x_value)


if __name__ == '__main__':
  tf.app.run(main)
