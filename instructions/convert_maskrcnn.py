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

import sys
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert as trt

config = tf.ConfigProto()
rewriter_config = config.graph_options.rewrite_options
rewriter_config.optimizers.extend([
    "constfold", "layout", "constfold", "arithmetic", "constfold", "arithmetic",
    "constfold"
])
rewriter_config.meta_optimizer_iterations = (
    rewriter_config_pb2.RewriterConfig.ONE)

saved_model_dir = sys.argv[1]
trt_saved_model_dir = sys.argv[2]
converter = trt.TrtGraphConverter(
    input_saved_model_dir=saved_model_dir,
    session_config=config,
    max_workspace_size_bytes=1 << 30,
    precision_mode="FP16",
    is_dynamic_op=True,
    use_function_backup=False)
converter.convert()
converter.save(trt_saved_model_dir)
