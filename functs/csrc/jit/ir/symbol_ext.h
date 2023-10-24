#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace c10 {
namespace immutable {

static const Symbol ns = Symbol::fromQualString("namespaces::immutable");

static const Symbol Assign = Symbol::fromQualString("immut::assign");
static const Symbol Update = Symbol::fromQualString("immut::access");

static auto Select = Symbol::fromQualString("immut::select");
static auto SelectReverse = Symbol::fromQualString("immut::select_rev");

static auto Slice = Symbol::fromQualString("immut::slice");
static auto SliceReverse = Symbol::fromQualString("immut::slice_rev");

} // namespace immutable
} // namespace c10
