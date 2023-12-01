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

static auto Permute = Symbol::fromQualString("immut::permute");
static auto PermuteRev = Symbol::fromQualString("immut::permute_rev");

static auto Expand = Symbol::fromQualString("immut::expand");
static auto ExpandRev = Symbol::fromQualString("immut::expand_rev");

static auto ExpandAs = Symbol::fromQualString("immut::expand_as");
static auto ExpandAsRev = Symbol::fromQualString("immut::expand_as_rev");

static auto Repeat = Symbol::fromQualString("immut::repeat");
static auto RepeatRev = Symbol::fromQualString("immut::repeat_rev");

static auto Index = Symbol::fromQualString("immut::index");
static auto IndexRev = Symbol::fromQualString("immut::index_rev");

// for fusion immutable operators
static auto Tensor = Symbol::fromQualString("immut::tensor");

static std::unordered_map<Symbol, Symbol> immutableVersion{
    {aten::copy_, c10::immutable::Assign},
    {aten::select, c10::immutable::Select},
    {aten::slice, c10::immutable::Slice},
    {aten::permute, c10::immutable::Permute},
    {aten::squeeze, c10::immutable::Squeeze},
    {aten::unsqueeze, c10::immutable::Unsqueeze},
#ifdef ENABLE_NNC_INT_LIST
    {aten::view, c10::immutable::View},
    {aten::reshape, c10::immutable::Reshape},
    {aten::expand, c10::immutable::Expand},
    {aten::expand_as, c10::immutable::ExpandAs},
    {aten::repeat, c10::immutable::Repeat},
    {aten::index, c10::immutable::Index},
#endif //  ENABLE_NNC_INT_LIST
};

static std::unordered_map<Symbol, Symbol> reverseVersion{
    {c10::immutable::Assign, c10::immutable::Assign},
    {c10::immutable::Select, c10::immutable::SelectReverse},
    {c10::immutable::Slice, c10::immutable::SliceReverse},
    {c10::immutable::Permute, c10::immutable::PermuteRev},
    {c10::immutable::Squeeze, c10::immutable::Unsqueeze},
    {c10::immutable::Unsqueeze, c10::immutable::Squeeze},
    {c10::immutable::View, c10::immutable::View},
    {c10::immutable::Reshape, c10::immutable::View},
    {c10::immutable::Expand, c10::immutable::ExpandRev},
    {c10::immutable::ExpandAs, c10::immutable::ExpandAsRev},
    {c10::immutable::Repeat, c10::immutable::RepeatRev},
    {c10::immutable::Index, c10::immutable::IndexRev},
};

} // namespace immutable
} // namespace c10

namespace torch {
namespace jit {

void init_immut_ops();

} // namespace jit
} // namespace torch
