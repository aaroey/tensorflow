from PIL import Image
from io import BytesIO
import numpy as np
import grpc
import requests
import time
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('model_name', 'mymodel',
                           'Name of the model being served')
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'Model server address')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size of each request')
tf.app.flags.DEFINE_boolean(
    'preprocess_image', True,
    'Whether to preprocess the input image before sending the request.')
FLAGS = tf.app.flags.FLAGS

IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
IMAGE_SIZE = 224
NUM_REQUESTS = 10

# These are the means of each channel that the input image data need to
# subtract.
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


def main(_):
  get_image_response = requests.get(IMAGE_URL)

  predict_request = predict_pb2.PredictRequest()
  predict_request.model_spec.name = FLAGS.model_name
  predict_request.model_spec.signature_name = 'predict'

  if FLAGS.preprocess_image:
    img = Image.open(BytesIO(get_image_response.content))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
    img_data = np.array(img.getdata()).reshape(img_shape).astype(np.float32)
    img_data -= _CHANNEL_MEANS
    img_data = img_data.repeat(FLAGS.batch_size, axis=0)
    predict_request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            img_data.astype(np.float32),
            shape=[FLAGS.batch_size] + img_shape[1:]))
  else:
    predict_request.inputs['image_bytes'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            get_image_response.content, shape=[1]))

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  # Warmup runs.
  for _ in range(5):
    response = stub.Predict(predict_request)

  t_start = time.time()
  for _ in range(NUM_REQUESTS):
    response = stub.Predict(predict_request)
  duration = time.time() - t_start

  predicted_label = np.array(response.outputs['classes'].int64_val)

  # The result should be 286 which is 'cougar, puma, catamount, mountain lion,
  # painter, panther, Felis concolor' accoridng to imagenet category.
  print('prediction result: ' + str(predicted_label))
  print('batch size: ' + str(FLAGS.batch_size))
  print('number of requests: ' + str(NUM_REQUESTS))
  print('duration (seconds) per request: ' + str(duration / NUM_REQUESTS))


if __name__ == '__main__':
  tf.app.run(main)
