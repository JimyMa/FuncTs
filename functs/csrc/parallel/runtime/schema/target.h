#pragma once

#include <cstddef>
#include <cstdint>

enum Target {
  kCPU = 1,
  kCUDA = 2,
  kROCM = 10,
};

struct DeviceContext final {
  explicit DeviceContext(Target target = kCPU, int device_id = 0)
      : target(target), device_id(device_id) {}
  Target target;
  int device_id;
};
