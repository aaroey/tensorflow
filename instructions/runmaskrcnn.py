# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
  input_saved_model_dir = "/tmp/maskrcnn"
  converter = trt.TrtGraphConverter(
      input_saved_model_dir=input_saved_model_dir,
      precision_mode="FP16",
      is_dynamic_op=True)
  converter.convert()
  trt_saved_model_dir = "/tmp/maskrcnn-trt"
  converter.save(trt_saved_model_dir)

  outputs, feed_dict = get_feeds_fetches()
  with tf.Graph().as_default():
    with tf.Session() as sess:
      loader.load(sess, [tag_constants.SERVING], trt_saved_model_dir)
      sess.run(outputs, feed_dict=feed_dict)


if __name__ == "__main__":
  tf.app.run(main)
