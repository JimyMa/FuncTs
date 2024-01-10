#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <string>

#include "functs/csrc/utils/logging.h"

namespace runtime {

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LONG_TAIL_ABORT(                                                      \
          "CUDA Driver Error: " << #x << " failed with error: " << msg  \
                                << std::endl);                          \
    }                                                                   \
  }

#define CUDA_CALL(func)                                                 \
  {                                                                     \
    cudaError_t e = (func);                                             \
    if (!(e == cudaSuccess || e == cudaErrorCudartUnloading)) {         \
      LONG_TAIL_ABORT(                                                      \
          "CUDA Runtime Error: " << cudaGetErrorString(e) << std::endl) \
    }                                                                   \
  }

#define NVRTC_CALL(x)                                                   \
  {                                                                     \
    nvrtcResult result = x;                                             \
    if (result != NVRTC_SUCCESS) {                                      \
      LONG_TAIL_ABORT(                                                      \
          "Nvrtc Error: " << #x << " failed with error: "               \
                          << nvrtcGetErrorString(result) << std::endl); \
    }                                                                   \
  }

} // namespace runtime
