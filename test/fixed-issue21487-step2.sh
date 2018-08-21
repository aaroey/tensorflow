#!/bin/bash
set -x
python -m tensorflow.python.tools.freeze_graph --input_graph log/graph.pbtxt --input_checkpoint log/model.ckpt-200 --output_node_names softmax_tensor --output_graph log/freeze_graph.pb
