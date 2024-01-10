#ifndef ELENA_INCLUDE_RUNTIME_FUNC_WRAPPER_H_
#define ELENA_INCLUDE_RUNTIME_FUNC_WRAPPER_H_

#include <string>

struct FuncConfig {
  int device_id;
  int gridDim_x;
  int gridDim_y;
  int gridDim_z;
  int blockDim_x;
  int blockDim_y;
  int blockDim_z;
  std::string func_name;
  std::string binary;
};
#endif  // ELENA_INCLUDE_RUNTIME_FUNC_WRAPPER_H_
