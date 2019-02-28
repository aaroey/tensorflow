import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

smdir = "/tmp/mobilenet"
mobilenet = tf.keras.applications.MobileNet()
tf.saved_model.save(mobilenet, smdir)

converter = trt.TrtGraphConverter(
    input_saved_model_dir=smdir,
    is_dynamic_op=True)
cf0 = converter.convert()

smdir_trt = smdir + "_trt1"
converter.save(smdir_trt)

inp = np.random.random_sample([1,224,224,3]).astype(np.float32)
inpconst = tf.constant(inp)
msgs = []
run_fn = lambda f: f(**{f._arg_keywords[0]: inpconst})

def time_fn(d, against=None):
  sm = tf.saved_model.load(d)
  cf = sm.signatures["serving_default"]
  NUM_RUNS = 100
  v = None
  for _ in range(2):  # warm up
    run_fn(cf)

  dt0 = datetime.datetime.now()
  for i in range(NUM_RUNS):
    v = run_fn(cf)
  dt1 = datetime.datetime.now()

  t = dt1 - dt0
  v = v[v.keys()[0]]

  msgs.append("------> time: %s" % str(t))
  if against is not None:
    msgs.append("------> max diff: %s" % str(np.max(np.abs(v - against))))
  return v

v = time_fn(smdir)
time_fn(smdir_trt, v)
for m in msgs:
  print(m)
