# To run this example, download this script to /tmp, then do:
#
# $ mkdir /tmp/tftrtresnet
# $ cd /tmp/tftrtresnet
# $ curl -O http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz
# $ tar zxf resnet_v2_fp32_savedmodel_NCHW.tar.gz
# $ docker run --runtime=nvidia --rm -v /tmp:/tmp -it \
#     tensorflow/tensorflow:1.12.0-gpu bash -c "
#     pip install requests Pillow;
#     python /tmp/resnet50_python_convert_and_run.py \
#         --input_saved_model_dir /tmp/tftrtresnet/resnet_v2_fp32_savedmodel_NCHW/1538687196 \
#         --output_saved_model_dir /tmp/tftrtresnet/resnet_v2_fp32_savedmodel_NCHW_trt/1538687196"

from PIL import Image
from io import BytesIO
import numpy as np
import requests
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants

tf.app.flags.DEFINE_string('input_saved_model_dir', '', '')
tf.app.flags.DEFINE_string('output_saved_model_dir', '', '')
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
  img_data = img_data.repeat(64, axis=0)  # 64 is batch size
  return img_data


def main(_):
  trt.create_inference_graph(
      None,
      None,
      max_batch_size=64,  # The official ResNet model is using batch size 64.
      precision_mode='FP32',
      is_dynamic_op=False,
      input_saved_model_dir=FLAGS.input_saved_model_dir,
      output_saved_model_dir=FLAGS.output_saved_model_dir)

  g = tf.Graph()
  with g.as_default():
    with tf.Session() as sess:
      loader.load(sess, [tag_constants.SERVING], FLAGS.output_saved_model_dir)
      result = sess.run(
          ['ArgMax:0', 'softmax_tensor:0'],
          feed_dict={'input_tensor:0': get_preprocessed_image_data()})
      # The result should be 286 which is 'cougar, puma, catamount, mountain
      # lion, painter, panther, Felis concolor' accoridng to imagenet category.
      print(result)


if __name__ == '__main__':
  tf.app.run(main)
