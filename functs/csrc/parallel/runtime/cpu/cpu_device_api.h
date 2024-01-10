#pragma once

#include <cstdlib>
#include <cstring>
#include <new>

#include "functs/csrc/utils/logging.h"

#include "functs/csrc/parallel/runtime/device_api.h"

namespace runtime {
class CPUDeviceAPI final : public DeviceAPI {
  void* AllocDataSpace(DeviceContext ctx, size_t nbytes, size_t alignment)
      final {
    void* ptr;
    // posix_memalign is available in android ndk since __ANDROID_API__ >= 17
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0)
      throw std::bad_alloc();
    return ptr;
  }

  void FreeDataSpace(DeviceContext ctx, void* ptr) final {
    free(ptr);
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
    memcpy(
        static_cast<char*>(to) + to_offset,
        static_cast<const char*>(from) + from_offset,
        size);
  }
  void print() final {
    LONG_TAIL_LOG_INFO("HELLO CPU API");
  }
};
} // namespace runtime
