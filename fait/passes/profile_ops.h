#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include "util/profile.h"

namespace c10 {
namespace prof {

static const Symbol ns = Symbol::fromQualString("namespaces::prof");
static const Symbol Begin = Symbol::fromQualString("prof::Begin");
static const Symbol End = Symbol::fromQualString("prof::End");

}  // namespace prof
}  // namespace c10

namespace torch {
namespace jit {

namespace prof = c10::prof;

/// @brief Convert all `prim::Print(str label, bool begin)` to profiling
/// instrumentation.
/// @param graph The graph to be processed.
void ConvertProfilingInstrumentation(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch