#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/symbol.h>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>

namespace c10 {
namespace tensorssa {

static const Symbol tssa = Symbol::fromQualString("namespaces::tssa");
static Symbol Update = Symbol::fromQualString("tssa::update");

} // namespace tensorssa
namespace immutable {

static const Symbol immut = Symbol::fromQualString("namespaces::immutable");

static Symbol Access = Symbol::fromQualString("immut::access");
static Symbol Assign = Symbol::fromQualString("immut::assign");

static auto Select = Symbol::fromQualString("immut::select");
static auto SelectReverse = Symbol::fromQualString("immut::select_rev");

static auto Slice = Symbol::fromQualString("immut::slice");
static auto SliceReverse = Symbol::fromQualString("immut::slice_rev");

static auto Squeeze = Symbol::fromQualString("immut::squeeze");
static auto Unsqueeze = Symbol::fromQualString("immut::unsqueeze");

static auto View = Symbol::fromQualString("immut::view");
static auto ViewRev = Symbol::fromQualString("immut::view_rev");

static auto Reshape = Symbol::fromQualString("immut::reshape");
static auto ReshapeRev = Symbol::fromQualString("immut::reshape_rev");

static std::unordered_map<Symbol, Symbol> immutableVersion{
    {aten::copy_, c10::immutable::Assign},
    {aten::select, c10::immutable::Select},
    {aten::slice, c10::immutable::Slice},
    {aten::squeeze, c10::immutable::Squeeze},
    {aten::unsqueeze, c10::immutable::Unsqueeze},
    {aten::view, c10::immutable::View},
    {aten::reshape, c10::immutable::Reshape},
};

static std::unordered_map<Symbol, Symbol> reverseVersion{
    {c10::immutable::Assign, c10::immutable::Assign},
    {c10::immutable::Select, c10::immutable::SelectReverse},
    {c10::immutable::Slice, c10::immutable::SliceReverse},
    {c10::immutable::Squeeze, c10::immutable::Unsqueeze},
    {c10::immutable::Unsqueeze, c10::immutable::Squeeze},
    {c10::immutable::View, c10::immutable::Unsqueeze},
    {c10::immutable::Reshape, c10::immutable::Squeeze}};

} // namespace immutable
} // namespace c10
