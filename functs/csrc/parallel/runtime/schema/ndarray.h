#pragma once

#include <memory>
#include <vector>

#include "functs/csrc/parallel/runtime/schema/dtype.h"
#include "functs/csrc/parallel/runtime/schema/target.h"
#include "functs/csrc/parallel/runtime/schema/tensor.h"

class NDArray {
 public:
  NDArray() {}
  void CopyFrom(NDArray& other);
  void CopyFromBytes(void* data, size_t len);
  static NDArray Empty(
      std::vector<size_t> shape,
      DataType dtype,
      DeviceContext ctx);
  std::shared_ptr<Tensor> tensor() {
    return tensor_;
  }
  void* data() {
    return tensor_->data();
  }
  DeviceContext ctx() {
    return tensor_->ctx();
  }
  std::vector<size_t>& shape() {
    return tensor_->shape();
  }
  DataType dtype() {
    return tensor_->dtype();
  }

 private:
  std::shared_ptr<Tensor> tensor_;
};
