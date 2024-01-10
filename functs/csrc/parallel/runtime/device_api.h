#pragma once

#include "functs/csrc/parallel/runtime/schema/target.h"
#include "functs/csrc/parallel/runtime/schema/tensor.h"
#include "functs/csrc/utils/logging.h"

/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = 128;

/*! \brief Number of bytes each allocation
must align to in temporary allocation */
constexpr int kTempAllocaAlignment = 128;

/*! \brief Maximum size that can be allocated on stack */
constexpr int kMaxStackAlloca = 1024;

namespace runtime {

inline const char* DeviceName(int type) {
  switch (type) {
    case kCPU:
      return "CPU";
    case kROCM:
      return "ROCM";
    case kCUDA:
      return "CUDA";
    default:
      LONG_TAIL_ABORT("unknown Device Type" << static_cast<int>(type));
  }
}

class DeviceAPI {
 public:
  virtual void print() = 0;
  /*!
   * \brief Allocate a data space on device.
   * \param ctx The device context to perform operation.
   * \param nbytes The number of bytes in memory.
   * \param alignment The alignment of the memory.
   * \return The allocated device pointer.
   */
  virtual void* AllocDataSpace(
      DeviceContext ctx,
      size_t nbytes,
      size_t alignment) = 0;
  /*!
   * \brief Free a data space on device.
   * \param ctx The device context to perform operation.
   * \param ptr The data space.
   */
  virtual void FreeDataSpace(DeviceContext ctx, void* ptr) = 0;
  /*!
   * \brief copy data from one place to another
   * \param from The source array.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param num_bytes The size of the memory in bytes
   * \param ctx_from The source context
   * \param ctx_to The target context
   * \param stream Optional stream object.
   */
  virtual void CopyDataFromTo(
      const void* from,
      size_t from_offset,
      void* to,
      size_t to_offset,
      size_t num_bytes,
      DeviceContext ctx_from,
      DeviceContext ctx_to,
      void* stream) = 0;
};

} // namespace runtime
