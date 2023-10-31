#pragma once

namespace torch {
namespace jit {

extern "C" bool metricsEnabled();

extern "C" void initializeMetrics();
extern "C" void finalizeMetrics();

extern "C" void beginProfilerPass();
extern "C" void endProfilerPass();
extern "C" bool allPassesSubmitted();

}  // namespace jit
}  // namespace torch