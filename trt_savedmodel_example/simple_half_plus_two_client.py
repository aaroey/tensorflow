import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('model_name', 'mymodel',
                           'Name of the model being served')
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'Model server address')
tf.app.flags.DEFINE_float('x_value', 1.0, 'x value in y =0.5 * x + 2')
FLAGS = tf.app.flags.FLAGS


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = 'serving_default'
  example = request.inputs['x'].CopyFrom(
      tf.contrib.util.make_tensor_proto([FLAGS.x_value]))
  response = stub.Predict(request)
  for key in response.outputs:
    print('prediction result: ' + key + ': ' + str(response.outputs[key]))


if __name__ == '__main__':
  tf.app.run(main)
