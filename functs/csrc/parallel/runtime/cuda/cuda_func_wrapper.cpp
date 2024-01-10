#include "functs/csrc/parallel/runtime/cuda/cuda_func_wrapper.h"

void CudaFuncWrapper::AnalyseArgs(NDArray array) {
  data_addrs.push_back(array.data());
}

void CudaFuncWrapper::AnalyseArgs(std::vector<NDArray> ndarray) {
  data_addrs.clear();
  for (size_t i = 0; i < ndarray.size(); i++) {
    data_addrs.push_back(ndarray[i].data());
  }
  // dynamic shape
  if (func_config.dyn_shape) {
    std::vector<void*> shape_dims;
    for (size_t i = 0; i < ndarray.size(); i++) {
      for (auto& shape_dim : ndarray[i].shape()) {
        shape_dims.push_back(&shape_dim);
      }
    }
    data_addrs.insert(data_addrs.end(), shape_dims.begin(), shape_dims.end());
  }
}

void CudaFuncWrapper::GetFunc() {
  cuModuleLoadDataEx(&module_, func_config.binary.c_str(), 0, 0, 0);
  CUDA_DRIVER_CALL(
      cuModuleGetFunction(&kernel_, module_, func_config.func_name.c_str()));
}

void CudaFuncWrapper::run() {
  std::mutex mutex_;
  std::lock_guard<std::mutex> lock(mutex_);

  if (kernel_ == nullptr) {
    GetFunc();
  }

  void* cudaargs[data_addrs.size()];
  for (size_t i = 0; i < data_addrs.size(); i++) {
    cudaargs[i] = &data_addrs[i];
  }

  CUDA_DRIVER_CALL(cuLaunchKernel(
      kernel_,
      func_config.gridDim_x,
      func_config.gridDim_y,
      func_config.gridDim_z,
      func_config.blockDim_x,
      func_config.blockDim_y,
      func_config.blockDim_z,
      0,
      nullptr,
      cudaargs,
      0));
  cuCtxSynchronize();
}
