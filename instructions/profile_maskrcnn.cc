#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

using absl::StrAppend;
using absl::StrCat;
using tensorflow::int32;
using tensorflow::string;

int64 FLAGS_num_threads = 1;
int64 FLAGS_num_requests = 100;
string FLAGS_model_dir = "";

string FLAGS_input_file = "";
int64 FLAGS_resize_to_width = 1472;
int64 FLAGS_resize_to_height = 896;

#define MYLOG LOG(INFO) << "------------------------> "

struct Options {
  int num_threads;
  int num_requests;
};

template <typename T>
Status ReadFromEnvVar(const string& name, T default_val, T* value);

template <>
Status ReadFromEnvVar(const string& name, bool default_val, bool* value) {
  return ReadBoolFromEnvVar(name, default_val, value);
}

template <>
Status ReadFromEnvVar(const string& name, int64 default_val, int64* value) {
  return ReadInt64FromEnvVar(name, default_val, value);
}

template <>
Status ReadFromEnvVar(const string& name, string default_val, string* value) {
  return ReadStringFromEnvVar(name, default_val, value);
}

template <typename T>
T GetEnvVar(const string& name, const T& default_val) {
  T value;
  int pos = name.find("FLAGS_");
  const string envvar = name.substr(pos + 6);
  TF_QCHECK_OK(ReadFromEnvVar(envvar, default_val, &value));
  LOG(INFO) << "Reading from " << envvar << " default: " << default_val
            << " vs actual: " << value;
  return value;
}

#define GetFlag(name) GetEnvVar(#name, name)

Tensor InputTensor(const Options& opts) {
  static mutex mu;
  static Tensor* input_tensor = nullptr;
  {
    tf_shared_lock lock(mu);
    if (input_tensor) return *input_tensor;
  }

  mutex_lock lock(mu);
  int original_width = 0, original_height = 0, components = 0;
  const int new_width = GetFlag(FLAGS_resize_to_width);
  const int new_height = GetFlag(FLAGS_resize_to_height);
  string data_str;
  TF_QCHECK_OK(
      ReadFileToString(Env::Default(), GetFlag(FLAGS_input_file), &data_str));
  MYLOG << "Input file size: " << data_str.size();
  {
    QCHECK(jpeg::GetImageInfo(data_str.c_str(),
                              data_str.size(),
                              &original_width,
                              &original_height,
                              &components));
    MYLOG << "input image size: " << original_width << "x" << original_height
          << "x" << components;
    QCHECK_GE(original_width, new_width);
    QCHECK_GE(original_height, new_height);
    QCHECK_EQ(3, components);
  }

  input_tensor =
      new Tensor(DT_FLOAT, TensorShape({1, new_height, new_width, 3}));
  {
    jpeg::UncompressFlags uncompress_flags;
    uncompress_flags.crop = true;
    uncompress_flags.crop_x = (original_width - new_width) / 2;
    uncompress_flags.crop_y = (original_height - new_height) / 2;
    uncompress_flags.crop_width = new_width;
    uncompress_flags.crop_height = new_height;
    int actual_width = 0, actual_height = 0, actual_components = 0;
    std::unique_ptr<uint8[]> data_uint8(jpeg::Uncompress(data_str.c_str(),
                                                         data_str.size(),
                                                         uncompress_flags,
                                                         &actual_width,
                                                         &actual_height,
                                                         &actual_components,
                                                         /*nwarn=*/nullptr));
    MYLOG << "Size after cropping: " << actual_width << "x" << actual_height
          << "x" << actual_components;
    QCHECK_EQ(actual_width, new_width);
    QCHECK_EQ(actual_height, new_height);
    QCHECK_EQ(actual_components, 3);

    float* data = input_tensor->flat<float>().data();
    for (int i = 0; i < input_tensor->NumElements(); ++i) {
      data[i] = static_cast<float>(data_uint8[i]) / 255.0f;
    }
  }
  return *input_tensor;
}  // namespace tensorflow

void MyComputeFn(Session* session, const Options& opts, int runs) {
  std::vector<Tensor> outputs;
  for (int i = 0; i < runs; ++i) {
    outputs.clear();
    TF_CHECK_OK(session->Run({{"Placeholder:0", InputTensor(opts)}},
                             {"Detections:0", "Sigmoid:0", "ImageInfo:0"},
                             {},
                             &outputs));
  }
}

std::unique_ptr<SavedModelBundle> CreateSessionFromSavedModel(
    SessionOptions* sess_options, const Options& opts) {
  std::unique_ptr<SavedModelBundle> bundle(new SavedModelBundle);
  TF_CHECK_OK(LoadSavedModel(*sess_options,
                             RunOptions(),
                             GetFlag(FLAGS_model_dir),
                             {"serve"},
                             bundle.get()));
  return bundle;
}

int64 ConcurrentSteps(Options opts) {
  // Creates a session.
  SessionOptions sess_options;
  MYLOG << "session config: \n" << sess_options.config.DebugString();

  std::unique_ptr<SavedModelBundle> bundle =
      CreateSessionFromSavedModel(&sess_options, opts);
  Session* session = bundle->session.get();

  MYLOG << "warming up...";
  MyComputeFn(session, opts, 3);
  MYLOG << "warm up complete, starting eval...";
  std::unique_ptr<thread::ThreadPool> step_threads;
  if (opts.num_threads > 1) {
    step_threads.reset(
        new thread::ThreadPool(Env::Default(), "trainer", opts.num_threads));
  }
  const int64 before = sess_options.env->NowMicros();
  if (opts.num_threads > 1) {
    for (int step = 0; step < opts.num_requests; ++step) {
      step_threads->Schedule(std::bind(&MyComputeFn, session, opts, 1));
    }
    step_threads.reset(nullptr);  // Wait for all threads to complete.
  } else {
    MyComputeFn(session, opts, opts.num_requests);
  }
  const int64 elapsed = sess_options.env->NowMicros() - before;
  return elapsed;
}

}  // end namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::Options opts;
  opts.num_threads = tensorflow::GetFlag(tensorflow::FLAGS_num_threads);
  opts.num_requests = tensorflow::GetFlag(tensorflow::FLAGS_num_requests);
  tensorflow::int64 elapsed = tensorflow::ConcurrentSteps(opts);
  MYLOG << "eval on "
        << opts.num_requests << " requests with " << opts.num_threads
        << " threads took " << elapsed << " us, = " << elapsed / 1000000.0
        << " seconds. Mean latency: " << elapsed / 1000.0 / opts.num_requests
        << " ms";
}
