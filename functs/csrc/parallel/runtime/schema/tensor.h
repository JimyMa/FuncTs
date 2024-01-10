#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "functs/csrc/parallel/runtime/device_api.h"
#include "functs/csrc/parallel/runtime/schema/dtype.h"
#include "functs/csrc/parallel/runtime/schema/target.h"

struct Tensor {
  Tensor() {}
  Tensor(std::vector<size_t>& shape, DataType dtype, DeviceContext ctx)
      : shape_(std::move(shape)), dtype_(dtype), ctx_(ctx) {}
  void* data() {
    return data_;
  }
  std::vector<size_t>& shape() {
    return shape_;
  }
  DataType dtype() {
    return dtype_;
  }
  DeviceContext ctx() {
    return ctx_;
  }
  size_t data_size();
  size_t align_size();
  void Allocate();
  void CopyFromBytes(void* data, size_t len);
  void CopyFrom(Tensor& other);
  ~Tensor();

 private:
  std::vector<size_t> shape_;
  DataType dtype_;
  DeviceContext ctx_;
  void* data_{nullptr};
};
