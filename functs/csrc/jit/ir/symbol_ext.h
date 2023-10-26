#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/symbol.h>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>

namespace c10 {
namespace immutable {

static const Symbol ns = Symbol::fromQualString("namespaces::immutable");

static Symbol Assign = Symbol::fromQualString("immut::assign");
static Symbol Update = Symbol::fromQualString("immut::access");

static auto Select = Symbol::fromQualString("immut::select");
static auto SelectReverse = Symbol::fromQualString("immut::select_rev");

static auto Slice = Symbol::fromQualString("immut::slice");
static auto SliceReverse = Symbol::fromQualString("immut::slice_rev");

static std::unordered_map<Symbol, Symbol> immutableVersion{
    {aten::copy_, c10::immutable::Assign},
    {aten::select, c10::immutable::Select},
    {aten::slice, c10::immutable::Slice}};

static std::unordered_map<Symbol, Symbol> inverseVersion{
    {c10::immutable::Assign, c10::immutable::Assign},
    {c10::immutable::Select, c10::immutable::SelectReverse},
    {c10::immutable::Slice, c10::immutable::SliceReverse}};

} // namespace immutable
} // namespace c10
