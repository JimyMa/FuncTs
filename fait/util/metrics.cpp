#include "metrics.h"

#include <cuda.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <nvperf_host.h>

#include <cstdio>
#include <string>

#include "cupti_ext/Eval.h"
#include "cupti_ext/FileOp.h"
#include "cupti_ext/Metric.h"

namespace torch {
namespace jit {

bool metricsEnabled() { return getenv("ENABLE_METRICS") != nullptr; }

static constexpr auto kMaxNumLaunches = 32000;

static std::vector<std::string> metricNames = {
    "sm__warps_active.avg.pct_of_peak_sustained_active",  // achieved_occupancy
    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",  // sm_efficiency
};

static const auto kRangeName = "Total";

#define EXIT_WAIVED 2

#define NVPW_API_CALL(apiFuncCall)                                         \
  do {                                                                     \
    NVPA_Status _status = apiFuncCall;                                     \
    if (_status != NVPA_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
              __FILE__, __LINE__, #apiFuncCall, _status);                  \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CUPTI_API_CALL(apiFuncCall)                                        \
  do {                                                                     \
    CUptiResult _status = apiFuncCall;                                     \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char* errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #apiFuncCall, errstr);                   \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define DRIVER_API_CALL(apiFuncCall)                                       \
  do {                                                                     \
    CUresult _status = apiFuncCall;                                        \
    if (_status != CUDA_SUCCESS) {                                         \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
              __FILE__, __LINE__, #apiFuncCall, _status);                  \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                         \
  do {                                                                        \
    cudaError_t _status = apiFuncCall;                                        \
    if (_status != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
              __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// Buffers
static std::vector<uint8_t> configImage, counterDataScratchBuffer,
    counterDataImage;

// Chip name
static std::string chipName;

// Parameters
static CUpti_Profiler_BeginPass_Params beginPassParams;
static CUpti_Profiler_EndPass_Params endPassParams;
static CUpti_Profiler_EnableProfiling_Params enableProfilingParams;
static CUpti_Profiler_DisableProfiling_Params disableProfilingParams;
static CUpti_Profiler_PushRange_Params pushRangeParams;
static CUpti_Profiler_PopRange_Params popRangeParams;

static bool createCounterDataImage(
    std::vector<uint8_t>& counterDataImagePrefix) {
  CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
  counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
  counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
  counterDataImageOptions.maxNumRanges = 1;
  counterDataImageOptions.maxNumRangeTreeNodes = 1;
  counterDataImageOptions.maxRangeNameLength = 64;

  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
      CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

  calculateSizeParams.pOptions = &counterDataImageOptions;
  calculateSizeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  CUPTI_API_CALL(
      cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

  CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
      CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  initializeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initializeParams.pOptions = &counterDataImageOptions;
  initializeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;

  counterDataImage.resize(calculateSizeParams.counterDataImageSize);
  initializeParams.pCounterDataImage = &counterDataImage[0];
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
      scratchBufferSizeParams = {
          CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratchBufferSizeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;
  scratchBufferSizeParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
      &scratchBufferSizeParams));

  counterDataScratchBuffer.resize(
      scratchBufferSizeParams.counterDataScratchBufferSize);

  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
      initScratchBufferParams = {
          CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  initScratchBufferParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;

  initScratchBufferParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  initScratchBufferParams.counterDataScratchBufferSize =
      scratchBufferSizeParams.counterDataScratchBufferSize;
  initScratchBufferParams.pCounterDataScratchBuffer =
      &counterDataScratchBuffer[0];

  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &initScratchBufferParams));

  return true;
}

void initializeMetrics() {
  CUdevice cuDevice;

  std::vector<uint8_t> counterDataImagePrefix;
  std::vector<uint8_t> counterAvailabilityImage;

  int deviceNum = 0;

  DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

  // Initialize profiler API and test device compatibility
  CUpti_Profiler_Initialize_Params profilerInitializeParams = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
  CUpti_Profiler_DeviceSupported_Params params = {
      CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
  params.cuDevice = deviceNum;
  CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

  if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
    ::std::cerr << "Unable to profile on device " << deviceNum << ::std::endl;
    exit(EXIT_WAIVED);
  }

  CUcontext cuContext;
  DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));

  /* Get chip name for the cuda device */
  CUpti_Device_GetChipName_Params getChipNameParams = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  getChipNameParams.deviceIndex = deviceNum;
  CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
  chipName = getChipNameParams.pChipName;

  CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  getCounterAvailabilityParams.ctx = cuContext;
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  counterAvailabilityImage.clear();
  counterAvailabilityImage.resize(
      getCounterAvailabilityParams.counterAvailabilityImageSize);
  getCounterAvailabilityParams.pCounterAvailabilityImage =
      counterAvailabilityImage.data();
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initializeHostParams = {
      NVPW_InitializeHost_Params_STRUCT_SIZE};
  NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

  if (!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage,
                                          counterAvailabilityImage.data())) {
    std::cout << "Failed to create configImage" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames,
                                                     counterDataImagePrefix)) {
    std::cout << "Failed to create counterDataImagePrefix" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!createCounterDataImage(counterDataImagePrefix)) {
    std::cout << "Failed to create counterDataImage" << std::endl;
    exit(EXIT_FAILURE);
  }

  CUpti_Profiler_BeginSession_Params beginSessionParams = {
      CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
  CUpti_Profiler_SetConfig_Params setConfigParams = {
      CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
  disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
  pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
  popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};

  beginSessionParams.ctx = NULL;
  beginSessionParams.counterDataImageSize = counterDataImage.size();
  beginSessionParams.pCounterDataImage = &counterDataImage[0];
  beginSessionParams.counterDataScratchBufferSize =
      counterDataScratchBuffer.size();
  beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
  beginSessionParams.range = CUPTI_UserRange;
  beginSessionParams.replayMode = CUPTI_UserReplay;
  beginSessionParams.maxRangesPerPass = 1;
  beginSessionParams.maxLaunchesPerPass = kMaxNumLaunches;

  CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

  setConfigParams.pConfig = &configImage[0];
  setConfigParams.configSize = configImage.size();

  setConfigParams.passIndex = 0;
  setConfigParams.minNestingLevel = 1;
  setConfigParams.numNestingLevels = 1;
  CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
  /* User takes the resposiblity of replaying the kernel launches */
  beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
  endPassParams = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
}

void finalizeMetrics() {
  CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
      CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
      CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
  CUpti_Profiler_EndSession_Params endSessionParams = {
      CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
  CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
  // DRIVER_API_CALL(cuCtxDestroy(cuContext));
  NV::Metric::Eval::PrintMetricValues(chipName, counterDataImage, metricNames);
}

void beginProfilerPass() {
  CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
  CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
  pushRangeParams.pRangeName = kRangeName;
  CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
}

void endProfilerPass() {
  CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));
  CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
  CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
}

bool allPassesSubmitted() { return endPassParams.allPassesSubmitted; }

}  // namespace jit
}  // namespace torch