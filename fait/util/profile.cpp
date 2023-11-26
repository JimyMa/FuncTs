#include "profile.h"

#include <c10/cuda/CUDAFunctions.h>
#include <cuda_profiler_api.h>
#include <cupti.h>
#include <nvToolsExt.h>

#include <chrono>
#include <iostream>

#include "util/common.h"
#include "util/ir.h"

namespace torch {
namespace jit {

using namespace std::chrono;

static bool cuptiEnabled() {
  return getenv("ENABLE_CUPTI") != nullptr;
}

static size_t totalAllocated = 0;
static size_t totalKernelLaunch = 0;

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE 8

static void recordMemory(CUpti_Activity* record) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMORY2: {
      auto memory = reinterpret_cast<CUpti_ActivityMemory3*>(record);
      if (memory->memoryOperationType !=
          CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION)
        return;
      totalAllocated += memory->bytes;
    } break;

    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      totalKernelLaunch++;
    } break;

    default:
      break;
  }
}

static void CUPTIAPI
bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *size = BUF_SIZE;
  *buffer = reinterpret_cast<uint8_t*>(aligned_alloc(ALIGN_SIZE, BUF_SIZE));
  *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  CUptiResult status;
  CUpti_Activity* record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        recordMemory(record);
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }
    } while (true);

    // report any records dropped from the queue
    size_t dropped;
    cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
    if (dropped != 0) {
      printf("Warning: Dropped %u activity records.\n", (unsigned int)dropped);
    }
  }

  free(buffer);
}

static void beginCuptiTrace() {
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
}

static void endCuptiTrace() {
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY2);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityFlushAll(1);
}

static bool enabled = false;

void enableProfiling() {
  cudaProfilerStart();
  if (cuptiEnabled())
    beginCuptiTrace();
  enabled = true;
}

void disableProfiling() {
  cudaProfilerStop();
  if (cuptiEnabled())
    endCuptiTrace();
  enabled = false;
}

struct TimeRecord {
  nanoseconds total{0}, min{INT64_MAX}, max{0};
  size_t count = 0;
  c10::optional<system_clock::time_point> begin = c10::nullopt;
  nvtxRangeId_t range;
};

static std::vector<std::string> labels;
static std::unordered_map<std::string, TimeRecord> records;

void profBegin(const std::string& label) {
  if (!enabled)
    return;
  at::cuda::device_synchronize();
  if (!records.count(label)) {
    labels.push_back(label);
    records.insert({label, {}});
  }
  records[label].begin = system_clock::now();
  records[label].range = nvtxRangeStartA(label.c_str());
}

void profEnd(const std::string& label) {
  if (!enabled)
    return;
  at::cuda::device_synchronize();
  auto& record = records.at(label);
  TORCH_CHECK(
      record.begin.has_value(), "`beginProfile` has not been called before.");
  nvtxRangeEnd(record.range);
  auto dur = system_clock::now() - *record.begin;
  record.begin = c10::nullopt;
  record.count++;
  record.total += dur;
  record.min = std::min(record.min, dur);
  record.max = std::max(record.max, dur);
}

static std::array<std::string, 4> timeUnits{"ns", "us", "ms", "s"};

std::string fmtDuration(nanoseconds dur) {
  double fp = dur.count();
  auto unitIdx = 0;
  while (unitIdx < timeUnits.size() - 1 && fp > 1e3) {
    fp /= 1e3;
    unitIdx++;
  }
  std::stringstream ss;
  ss << std::setprecision(4) << fp << timeUnits[unitIdx];
  return ss.str();
}

static std::array<std::string, 4> byteUnits{"B", "KB", "MB", "GB"};

std::string fmtBytes(size_t bytes) {
  double fp = bytes;
  auto unitIdx = 0;
  while (unitIdx < byteUnits.size() - 1 && fp > 1024) {
    fp /= 1024;
    unitIdx++;
  }
  std::stringstream ss;
  ss << std::setprecision(4) << fp << byteUnits[unitIdx];
  return ss.str();
}

static constexpr auto kLabelWidth = 16;
static constexpr auto kStatWidth = 10;

static void printLabel(const std::string& label) {
  print(
      std::cout,
      std::setw(kLabelWidth),
      std::setiosflags(std::ios::left),
      label,
      std::resetiosflags(std::ios::left));
}

template <class T>
static void printStat(T&& stat) {
  print(std::cout, std::setw(kStatWidth), stat);
}

void printProfilingResults(size_t count) {
  // Print CUPTI results
  if (totalKernelLaunch > 0)
    std::cout << "Kernel launch: " << totalKernelLaunch / count << '\n';
  if (totalAllocated > 0)
    std::cout << "Memory allocation: " << fmtBytes(totalAllocated / count)
              << '\n';

  if (records.empty())
    return;

  // Print ranges
  std::cout << "\nRanges:\n";
  printLabel("Label");
  printStat("Count");
  printStat("Total");
  printStat("Mean");
  printStat("Min");
  printStat("Max");
  std::cout << '\n';

  for (auto& label : labels) {
    auto& record = records[label];
    printLabel(label);
    printStat(record.count); // count
    if (record.count == 0) {
      std::cout << '\n';
      continue;
    }
    printStat(fmtDuration(record.total)); // total
    printStat(fmtDuration(record.total / record.count)); // mean
    printStat(fmtDuration(record.min)); // min
    printStat(fmtDuration(record.max)); // max
    std::cout << '\n';
  }
}

} // namespace jit
} // namespace torch