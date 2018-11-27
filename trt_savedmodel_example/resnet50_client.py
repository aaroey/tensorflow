from PIL import Image
from io import BytesIO
import base64
import grpc
import numpy as np
import requests
import time
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('model_name', 'mymodel',
                           'Name of the model being served')
tf.app.flags.DEFINE_string('grpc_server', 'localhost:8500',
                           'Model server address for gRPC connection')
tf.app.flags.DEFINE_string('rest_api_server',
                           'http://localhost:8501/v1/models/mymodel:predict',
                           'Model server address for RESTful connection')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size of each request')
tf.app.flags.DEFINE_boolean('use_rest_api', False,
                            'Whether to use REST API or gRPC')
tf.app.flags.DEFINE_boolean(
    'preprocess_image', True,
    'Whether to preprocess the input image before sending the request.')
FLAGS = tf.app.flags.FLAGS

IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
NUM_REQUESTS = 10
IMAGE_SIZE = 224
IMAGE_SHAPE = [1, IMAGE_SIZE, IMAGE_SIZE, 3]


def get_raw_image_data():
  get_image_response = requests.get(IMAGE_URL)
  return get_image_response.content


def get_preprocessed_image_data():
  # These are the means of each channel that the input image data need to
  # subtract.
  _R_MEAN = 123.68
  _G_MEAN = 116.78
  _B_MEAN = 103.94
  _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

  img = Image.open(BytesIO(get_raw_image_data()))
  img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
  img_data = np.array(img.getdata()).reshape(IMAGE_SHAPE).astype(np.float32)
  img_data -= _CHANNEL_MEANS
  img_data = img_data.astype(np.float32)
  img_data = img_data.repeat(FLAGS.batch_size, axis=0)
  return img_data


def send_request_with_grpc():
  predict_request = predict_pb2.PredictRequest()
  predict_request.model_spec.name = FLAGS.model_name
  predict_request.model_spec.signature_name = 'predict'

  if FLAGS.preprocess_image:
    predict_request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            get_preprocessed_image_data(),
            shape=[FLAGS.batch_size] + IMAGE_SHAPE[1:]))
  else:
    predict_request.inputs['image_bytes'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            get_raw_image_data(), shape=[FLAGS.batch_size]))

  channel = grpc.insecure_channel(FLAGS.grpc_server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  # Warmup runs.
  for _ in range(5):
    response = stub.Predict(predict_request)

  t_start = time.time()
  for _ in range(NUM_REQUESTS):
    response = stub.Predict(predict_request)
  elapsed_seconds = time.time() - t_start

  predicted_label = np.array(response.outputs['classes'].int64_val)
  return predicted_label, elapsed_seconds


def send_request_with_rest_api():
  predict_request = ''
  if FLAGS.preprocess_image:
    predict_request = '{"instances" : %s }' % (
        get_preprocessed_image_data().tolist())
  else:
    predict_request = '{"instances" : [{"b64": "%s"}]}' % base64.b64encode(
        get_raw_image_data())

  # Warmup runs.
  for _ in range(5):
    response = requests.post(FLAGS.rest_api_server, data=predict_request)
    response.raise_for_status()

  # Send few actual requests and report average latency.
  elapsed_seconds = 0
  for _ in range(NUM_REQUESTS):
    response = requests.post(FLAGS.rest_api_server, data=predict_request)
    response.raise_for_status()
    elapsed_seconds += response.elapsed.total_seconds()
    predicted_label = [
        item['classes'] for item in response.json()['predictions']
    ]

  return predicted_label, elapsed_seconds


def main(_):
  predicted_label, elapsed_seconds = (
      send_request_with_rest_api()
      if FLAGS.use_rest_api else send_request_with_grpc())

  # The result should be 286 which is 'cougar, puma, catamount, mountain lion,
  # painter, panther, Felis concolor' accoridng to imagenet category.
  print('prediction result: ' + str(predicted_label))
  print('batch size: ' + str(FLAGS.batch_size))
  print('number of requests: ' + str(NUM_REQUESTS))
  print('elapsed seconds per request: ' + str(elapsed_seconds / NUM_REQUESTS))


if __name__ == '__main__':
  tf.app.run(main)
