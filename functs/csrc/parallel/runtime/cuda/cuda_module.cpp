#include "functs/csrc/parallel/runtime/cuda/cuda_module.h"

#include <nvrtc.h>

#include "functs/csrc/parallel/runtime/cuda/cuda_common.h"
#include "functs/csrc/utils/logging.h"

void CudaModule::NVRTCCompile(
    const std::vector<std::string>& options = std::vector<std::string>()) {
  const std::string code = data;
  std::vector<std::string> compile_params;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;
  std::string cc = "30";
  int major, minor;
  cudaError_t e1 =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LONG_TAIL_LOG_INFO(
        "cannot detect compute capability from your device, "
        "fall back to compute_30.");
  }

  compile_params.push_back("-arch=compute_" + cc);
  compile_params.push_back("--fmad=false");

  for (const auto& string : compile_params) {
    param_cstrings.push_back(string.c_str());
  }

  for (const auto& string : options) {
    param_cstrings.push_back(string.c_str());
  }

  NVRTC_CALL(
      nvrtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  NVRTC_CALL(
      nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data()));

  size_t log_size;
  NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
  size_t ptx_size;
  NVRTC_CALL(nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CALL(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CALL(nvrtcDestroyProgram(&prog));

  data = ptx;
  fmt = "ptx";
}

CudaFuncWrapper CudaModule::GetFunction(
    std::string func_name,
    int gridDim_x,
    int gridDim_y,
    int gridDim_z,
    int blockDim_x,
    int blockDim_y,
    int blockDim_z) {
  int device_id = 0;
  std::string ptx;

  if (fmt == "source") {
    NVRTCCompile();
    ptx = data;
  } else if (fmt == "ptx") {
    ptx = data;
  } else {
    LONG_TAIL_ABORT("Only ptx and source are supported");
  }

  // CUDA_DRIVER_CALL(cuModuleLoadDataEx(&module, ptx.c_str(), 0, 0, 0));

  CudaFuncConfig config;
  config.device_id = device_id;
  config.gridDim_x = gridDim_x;
  config.gridDim_y = gridDim_y;
  config.gridDim_z = gridDim_z;
  config.blockDim_x = blockDim_x;
  config.blockDim_y = blockDim_y;
  config.blockDim_z = blockDim_z;
  config.func_name = func_name;
  config.binary = ptx;
  config.dyn_shape = dyn_shape;

  return CudaFuncWrapper(config);
}

CudaModule CudaModuleCreate(
    std::string data,
    std::string fmt,
    // const std::vector<std::pair<int,
    // ir::ScalarType>>& arg_list,
    std::string source,
    std::string assembly) {
  auto n = CudaModule(data, fmt, source, assembly);
  return n;
}
