load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

# bazel run -c opt --config=cuda maskrcnn:profile_maskrcnn_cc
tf_cc_binary(
    name = "profile_maskrcnn_cc",
    srcs = ["profile_maskrcnn.cc"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/contrib/tensorrt:trt_engine_op_op_lib",
        "//tensorflow/contrib/tensorrt:trt_op_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)
