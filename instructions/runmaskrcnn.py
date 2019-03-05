import requests
import io

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt


def get_feeds_fetches():
  url = "https://tensorflow.org/images/blogs/serving/cat.jpg"
  img = requests.get(url).content
  img = Image.open(io.BytesIO(img))
  img = img.resize((1472, 896), Image.ANTIALIAS)
  inp = np.array(img.getdata()).reshape(1, img.size[0], img.size[1], 3)
  inp = inp.transpose(0, 2, 1, 3)
  inp = (inp / 255.0).astype(dtype=np.float32)
  print("-----------------------> input shape: %s" % str(inp.shape))

  outputs = ["Detections:0", "Sigmoid:0", "ImageInfo:0"]
  feed_dict = {"Placeholder:0": inp}
  return outputs, feed_dict


num_runs = 100


def main(argv):
  input_saved_model_dir = "/tmp/model-nvidia-nms"
  converter = trt.TrtGraphConverter(
      input_saved_model_dir=input_saved_model_dir,
      precision_mode="FP16",
      is_dynamic_op=True)
  converter.convert()
  trt_saved_model_dir = "/tmp/mobilenet.trt"
  converter.save(trt_saved_model_dir)

  outputs, feed_dict = get_feeds_fetches()
  with tf.Graph().as_default():
    with tf.Session() as sess:
      loader.load(sess, [tag_constants.SERVING], trt_saved_model_dir)
      sess.run(outputs, feed_dict=feed_dict)


if __name__ == "__main__":
  tf.app.run(main)
