#pragma once

#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>

#include <chrono>
#include <iomanip>

#include "passes/profile_ops.h"
#include "util/common.h"
#include "util/metrics.h"

namespace torch {
namespace jit {

using namespace std::chrono;
using namespace std::chrono_literals;

template <class T>
inline T loadPickle(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  TORCH_CHECK(ifs, "Cannot open file ", path);
  std::vector<char> buf((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
  return torch::pickle_load(buf).to<T>();
}

inline IValue processIValue(const IValue &val) {
  if (val.isList()) {
    auto list = val.toListRef();
    c10::impl::GenericList newList(list.front().type());
    for (auto &elem : list) newList.push_back(processIValue(elem));
    return std::move(newList);
  } else if (val.isTuple()) {
    auto &tuple = val.toTupleRef().elements();
    std::vector<IValue> newValues;
    for (auto &elem : tuple) newValues.push_back(processIValue(elem));
    return c10::ivalue::Tuple::create(std::move(newValues));
  } else if (val.isTensor()) {
    return val.toTensor().cuda();
  } else
    return val;
}

inline Stack getFeatureSample(const c10::List<IValue> &dataset, size_t index) {
  auto tup = dataset.get(index).toTupleRef().elements();
  Stack inputs;
  inputs.push_back({});
  for (auto &val : tup) inputs.push_back(processIValue(val));
  return std::move(inputs);
}

struct EvalResult {
  nanoseconds total;
  size_t count = 0;

  nanoseconds mean() const { return total / int64_t(count); }
};

static constexpr auto kWarmupRuns = 16;
static constexpr auto kRunDuration = 10s;

inline void evalMetrics(const std::function<void(size_t)> &task,
                        size_t numSamples) {
  // Initialize
  initializeMetrics();

  // Warm up
  for (auto i : c10::irange(kWarmupRuns)) task(i);
  at::cuda::device_synchronize();

  // Run and replay
  do {
    beginProfilerPass();
    for (auto i : c10::irange(numSamples)) {
      task(i);
      at::cuda::device_synchronize();
    }
    endProfilerPass();
  } while (!allPassesSubmitted());

  // Print final results
  finalizeMetrics();
}

inline EvalResult evaluate(const std::function<void(size_t)> &task) {
  // Warm up
  for (auto i : c10::irange(kWarmupRuns)) task(i);
  at::cuda::device_synchronize();

  // Run for the expected period
  enableProfiling();
  size_t count = 0;
  auto begin = system_clock::now();
  while (system_clock::now() - begin < kRunDuration) {
    task(count++);
    at::cuda::device_synchronize();
  }
  disableProfiling();

  return {system_clock::now() - begin, count};
}

}  // namespace jit
}  // namespace torch