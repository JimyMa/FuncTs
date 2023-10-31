//
// Created by jimyma on 1/28/23.
//

#ifndef LONG_TAIL_CODEGEN_H
#define LONG_TAIL_CODEGEN_H

#include <vector>

#include "c10/core/ScalarType.h"

namespace torch {
namespace jit {

class KernelGen {
 public:
  class CallArg;
};

class CallArg {
 public:
  template <typename T>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-pro-type-const-cast)
  CallArg(const std::vector<T>& buffer)
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      : data_(const_cast<T*>(buffer.data())) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CallArg(void* ptr) : data_(ptr) {}

#define ARG_TYPE_CTOR(Type, Name) \
  CallArg(Type v) { memcpy(&data_, &v, sizeof(Type)); }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ARG_TYPE_CTOR);
#undef ARG_TYPE_CTOR

  void* data() const { return data_; }

#define ARG_PTR_DEFINE(Type, Name) \
  Type* Name##Ptr() const { return (Type*)&data_; }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ARG_PTR_DEFINE);
#undef ARG_PTR_DEFINE

 private:
  void* data_;
};
}  // namespace jit
}  // namespace torch
#endif  // LONG_TAIL_CODEGEN_H
