import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('model_name', 'mymodel',
                           'Name of the model being served')
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'Model server address')
FLAGS = tf.app.flags.FLAGS


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  input_shape = [128, 224, 224, 3]

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = 'predict'
  request.inputs['input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(
          # Synthetic inputs.
          np.random.uniform(input_shape).astype(np.float32),
          shape=input_shape))
  response = stub.Predict(request)
  predicted_label = np.array(response.outputs['classes'].int64_val)
  print('prediction result: ' + str(predicted_label))


if __name__ == '__main__':
  tf.app.run(main)
