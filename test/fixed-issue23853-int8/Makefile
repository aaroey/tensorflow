.PHONY: all

tf_src_dir = ../..
tf_bazel_bin_dir = $(tf_src_dir)/bazel-bin/tensorflow

all: main graph.pb
	LD_LIBRARY_PATH=$(tf_bazel_bin_dir):$(tf_bazel_bin_dir)/compiler/tf2tensorrt/python/ops:$(HOME)/Downloads/TensorRT-5.1.2.1-cuda-10.0/lib:$(HOME)/Downloads/TensorRT-5.0.2.6-cuda-9.0/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64: ./main

main: main.cpp session_config.h
	g++ -o main -std=c++17 -pthread main.cpp -I$(tf_src_dir)/ -L$(tf_bazel_bin_dir) -L$(tf_bazel_bin_dir)/compiler/tf2tensorrt/python/ops -ltensorflow -ltftrt

graph.pb: make_graph.py
	./make_graph.py

session_config.h: make_session_config.py
	./make_session_config.py > $@
