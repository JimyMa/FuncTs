#pragma once

#include <string>
#include <vector>

#include "functs/csrc/parallel/runtime/cuda/cuda_func_wrapper.h"
#include "functs/csrc/parallel/runtime/module.h"

class CudaModule final : public Module {
 public:
  CudaModule(
      std::string data,
      std::string fmt,
      std::string hipsource = "",
      std::string assembly = "",
      bool dyn_shape = false)
      : data(data),
        fmt(fmt),
        source(assembly),
        assembly(assembly),
        dyn_shape(dyn_shape) {}

  CudaFuncWrapper GetFunction(
      std::string func_name,
      int gridDim_x,
      int gridDim_y,
      int gridDim_z,
      int blockDim_x,
      int blockDim_y,
      int blockDim_z);

  void NVRTCCompile(const std::vector<std::string>& options);

 private:
  std::string data;
  std::string fmt;
  std::string source;
  std::string assembly;
  bool dyn_shape;
};
CudaModule CudaModuleCreate(
    std::string data,
    std::string fmt,
    // const std::vector<std::pair<int,
    // ir::ScalarType>>& arg_list,
    std::string source,
    std::string assembly);
