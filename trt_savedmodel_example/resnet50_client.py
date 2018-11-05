from PIL import Image
from io import BytesIO
import numpy as np
import grpc
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('model_name', 'mymodel',
                           'Name of the model being served')
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'Model server address')
FLAGS = tf.app.flags.FLAGS

IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
IMAGE_SIZE = 224


# These are the means of each channel that the input image data need to
# subtract.
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


def main(_):
  response = requests.get(IMAGE_URL)
  img = Image.open(BytesIO(response.content))
  img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
  input_shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
  img_data = np.array(img.getdata()).reshape(input_shape).astype(np.float32)
  img_data -= _CHANNEL_MEANS

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = 'predict'
  request.inputs['input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(
          img_data.astype(np.float32), shape=input_shape))
  response = stub.Predict(request)
  predicted_label = np.array(response.outputs['classes'].int64_val)[0]

  # The result should be 286 which is 'cougar, puma, catamount, mountain lion,
  # painter, panther, Felis concolor' accoridng to imagenet category.
  print('prediction result: ' + str(predicted_label))


if __name__ == '__main__':
  tf.app.run(main)
