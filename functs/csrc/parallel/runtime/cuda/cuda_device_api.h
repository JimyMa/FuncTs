#pragma once

#include <cstring>

#include "assert.h"
#include "functs/csrc/parallel/runtime/cuda/cuda_common.h"
#include "functs/csrc/parallel/runtime/device_api.h"
#include "functs/csrc/utils/logging.h"

namespace runtime {
class CUDADeviceAPI final : public DeviceAPI {
  void print() final {
    LONG_TAIL_LOG_INFO("HELLO CUDA API");
  }
  void* AllocDataSpace(DeviceContext ctx, size_t nbytes, size_t alignment)
      final {
    LONG_TAIL_ASSERT(256 % alignment == 0U, "CUDA space is aligned at 256 bytes");
    void* ret;
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaError_t e = cudaMalloc(&ret, nbytes);
    if (e == cudaErrorMemoryAllocation) {
      size_t free_size;
      size_t total_size;
      cudaMemGetInfo(&free_size, &total_size);
      LONG_TAIL_ABORT(
          cudaGetErrorString(e)
          << ": [Detail] You attempt to malloc " << nbytes
          << " bit data, but only " << total_size << "bit left");
    }
    // Use Cuda Unified Memory
    CUDA_CALL(cudaMallocManaged(&ret, nbytes));
    return ret;
  }

  void FreeDataSpace(DeviceContext ctx, void* ptr) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaFree(ptr));
  }

  void CopyDataFromTo(
      const void* from,
      size_t from_offset,
      void* to,
      size_t to_offset,
      size_t size,
      DeviceContext ctx_from,
      DeviceContext ctx_to,
      void* stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    // In case there is a copy from host mem to host mem */
    if (ctx_to.target == kCPU && ctx_from.target == kCPU) {
      memcpy(to, from, size);
      return;
    }

    if (ctx_from.target == kCUDA && ctx_to.target == kCUDA) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(
            to, ctx_to.device_id, from, ctx_from.device_id, size, cu_stream);
      }
    } else if (ctx_from.target == kCUDA && ctx_to.target == kCPU) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (ctx_from.target == kCPU && ctx_to.target == kCUDA) {
      CUDA_CALL(cudaSetDevice(ctx_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LONG_TAIL_ABORT("expect copy from/to GPU or between GPU");
    }
  }

 private:
  static void GPUCopy(
      const void* from,
      void* to,
      size_t size,
      cudaMemcpyKind kind,
      cudaStream_t stream) {
    if (stream != 0) {
      CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
      CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
  }
};
} // namespace runtime
