#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex> // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "functs/csrc/parallel/runtime/cuda/cuda_common.h"
#include "functs/csrc/parallel/runtime/schema/ndarray.h"

struct CudaFuncConfig {
  int device_id;
  int gridDim_x;
  int gridDim_y;
  int gridDim_z;
  int blockDim_x;
  int blockDim_y;
  int blockDim_z;
  std::string func_name;
  std::string binary;
  bool dyn_shape;
};

class CudaFuncWrapper {
 public:
  CudaFuncWrapper() : kernel_(nullptr) {}
  explicit CudaFuncWrapper(CudaFuncConfig func_config)
      : module_(nullptr), kernel_(nullptr), func_config(func_config) {}

  template <typename... Args>
  void operator()(Args&&... args);

  void run();
  template <typename... Args>
  void AnalyseArgs(NDArray ndarray, Args&&... args);
  void AnalyseArgs(NDArray ndarray);
  void AnalyseArgs(std::vector<NDArray> ndarrays);
  void GetFunc();

  // destructor
  ~CudaFuncWrapper() {
    if (module_ != nullptr) {
      CUDA_CALL(cudaSetDevice(func_config.device_id));
      CUDA_DRIVER_CALL(cuModuleUnload(module_));
    }
  }

 private:
  CUmodule module_;
  CUfunction kernel_;
  CudaFuncConfig func_config;
  std::vector<void*> data_addrs;
};

template <typename... Args>
void CudaFuncWrapper::operator()(Args&&... args) {
  data_addrs.clear();
  AnalyseArgs(std::forward<Args>(args)...);
  run();
}

template <typename... Args>
void CudaFuncWrapper::AnalyseArgs(NDArray array, Args&&... args) {
  data_addrs.push_back(array.data());
  AnalyseArgs(std::forward<Args>(args)...);
}
