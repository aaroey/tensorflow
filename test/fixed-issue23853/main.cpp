#include <tensorflow/c/c_api.h>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "session_config.h"

struct GraphDeleter {
  void operator()(TF_Graph *graph) const {
    if (graph) {
      TF_DeleteGraph(graph);
    }
  }
};

using GraphPtr = std::unique_ptr<TF_Graph, GraphDeleter>;

struct ImportGraphDefOptionsDeleter {
  void operator()(TF_ImportGraphDefOptions *options) const {
    if (options) {
      TF_DeleteImportGraphDefOptions(options);
    }
  }
};

using ImportGraphDefOptionsPtr = std::unique_ptr<
  TF_ImportGraphDefOptions, ImportGraphDefOptionsDeleter>;

struct StatusDeleter {
  void operator()(TF_Status *status) const {
    if (status) {
      TF_DeleteStatus(status);
    }
  }
};

using StatusPtr = std::unique_ptr<TF_Status, StatusDeleter>;

struct TensorDeleter {
  void operator()(TF_Tensor *tensor) const {
    if (tensor) {
      TF_DeleteTensor(tensor);
    }
  }
};

using TensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

GraphPtr LoadGraphDef(const char *path) {
  std::ifstream f(path);
  assert(f);
  f.seekg(0, std::ios_base::end);
  auto size = f.tellg();
  f.seekg(0, std::ios_base::beg);
  std::unique_ptr<char[]> data(new char[size]);
  f.read(data.get(), size);
  assert(f);

  TF_Buffer buffer;
  buffer.data = data.get();
  buffer.length = size;

  GraphPtr graph(TF_NewGraph());
  assert(graph);

  ImportGraphDefOptionsPtr options(TF_NewImportGraphDefOptions());
  assert(options);

  StatusPtr status(TF_NewStatus());
  assert(status);

  TF_GraphImportGraphDef(graph.get(), &buffer, options.get(), status.get());
  std::cout << TF_Message(status.get()) << std::endl;
  assert(TF_GetCode(status.get()) == TF_OK);

  return graph;
}

TF_Output GetOutput(TF_Graph *graph, const char *operation_name) {
  auto op = TF_GraphOperationByName(graph, operation_name);
  assert(op);
  return TF_Output{op, 0};
}

struct SessionDeleter {
  void operator()(TF_Session *session) const {
    if (session) {
      StatusPtr status(TF_NewStatus());
      assert(status);

      TF_DeleteSession(session, status.get());
      assert(TF_GetCode(status.get()) == TF_OK);
    }
  }
};

using SessionPtr = std::unique_ptr<TF_Session, SessionDeleter>;

struct SessionOptionsDeleter {
  void operator()(TF_SessionOptions *options) const {
    if (options) {
      TF_DeleteSessionOptions(options);
    }
  }
};

using SessionOptionsPtr = std::unique_ptr<TF_SessionOptions, SessionOptionsDeleter>;

SessionPtr MakeSession(TF_Graph *graph) {
  SessionOptionsPtr options(TF_NewSessionOptions());
  assert(options);

  StatusPtr status(TF_NewStatus());
  assert(status);

  TF_SetConfig(options.get(), session_config, sizeof(session_config), status.get());

  SessionPtr session(TF_NewSession(graph, options.get(), status.get()));
  assert(TF_GetCode(status.get()) == TF_OK);
  assert(session);

  return session;
}

int main() {
  auto graph = LoadGraphDef("graph.pb");
  assert(graph);

  auto session = MakeSession(graph.get());

  const std::array<TF_Output, 1> inputs{GetOutput(graph.get(), "input")};
  const std::array<TF_Output, 1> outputs{GetOutput(graph.get(), "output")};

  const std::array<std::int64_t, 4> dims{1, 256, 256, 3};
  const auto value = std::unique_ptr<float[]>(new float[dims[0] * dims[1] * dims[2] * dims[3]]);

  std::array<TensorPtr, 1> input_tensors{TensorPtr(TF_NewTensor(
    TF_FLOAT,
    /* dims */ &dims[0], dims.size(),
    /* data */ const_cast<float *>(value.get()), sizeof(float) * dims[0] * dims[1] * dims[2] * dims[3],
    /* deallocator */ [](void *, size_t, void *){},
    /* deallocator_arg */ nullptr))};

  std::vector<std::thread> threads;

  for (int i = 0; i < 3; ++i) {
    threads.emplace_back([&](){
      StatusPtr status(TF_NewStatus());
      assert(status);

      //while (true) {
      for (int j = 0; j < 10; ++j) {
        std::array<TensorPtr, 1> output_tensors;

        TF_SessionRun(session.get(),
          /* run_options */ nullptr,
          &inputs[0], reinterpret_cast<TF_Tensor **>(&input_tensors[0]), inputs.size(),
          &outputs[0], reinterpret_cast<TF_Tensor **>(&output_tensors[0]), outputs.size(),
          /* target_opers */ nullptr, 0,
          /* run_metadata */ nullptr,
          status.get());

        if (TF_GetCode(status.get()) != TF_OK) {
          std::cerr << "TF_SessionRun: " << TF_Message(status.get()) << std::endl;
          std::abort();
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}
