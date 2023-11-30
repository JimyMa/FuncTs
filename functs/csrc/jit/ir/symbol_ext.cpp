#include "functs/csrc/jit/ir/symbol_ext.h"
#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
at::Tensor Access(at::Tensor src) {
  return src;
}

at::Tensor Assign(at::Tensor self, at::Tensor src, c10::optional<bool> n) {
  return src;
}

at::Tensor ImmutSelect(at::Tensor src, int64_t dim, int64_t index) {
  return src.select(dim, index);
}

at::Tensor ImmutSelectRev(
    at::Tensor self,
    at::Tensor src,
    int64_t dim,
    int64_t index) {
  auto immut_self = self;
  immut_self.select(dim, index).copy_(src);
  return immut_self;
}

at::Tensor ImmutSlice(
    at::Tensor src,
    int64_t dim,
    c10::optional<int64_t> start = 0,
    c10::optional<int64_t> end = 0,
    int64_t step = 1) {
  return src.slice(dim, start, end, step);
}

at::Tensor ImmutSliceRev(
    at::Tensor self,
    at::Tensor src,
    int64_t dim,
    c10::optional<int64_t> start = 0,
    c10::optional<int64_t> end = 0,
    int64_t step = 1) {
  auto immut_self = self;
  immut_self.slice(dim, start, end, step).copy_(src);
  return immut_self;
}

at::Tensor ImmutSqueeze(at::Tensor src, int64_t dim) {
  return src.squeeze(dim);
}

at::Tensor ImmutUnqueeze(at::Tensor src, int64_t dim) {
  return src.unsqueeze(dim);
}

at::Tensor ImmutView(at::Tensor src, at::IntArrayRef dims) {
  return src.view(dims);
}

at::Tensor ImmutReshape(at::Tensor src, at::IntArrayRef dims) {
  return src.reshape(dims);
}

at::Tensor ImmutPermute(at::Tensor src, at::IntArrayRef dims) {
  return src.permute(dims);
}

at::Tensor ImmutPermuteRev(at::Tensor src, at::IntArrayRef dims) {
  return src.permute(dims);
}

at::Tensor ImmutExpand(
    at::Tensor src,
    at::IntArrayRef sizes,
    bool implicit = false) {
  return src.expand(sizes, implicit);
}

at::Tensor ImmutExpandRev(
    at::Tensor src,
    at::IntArrayRef sizes,
    bool implicit = false) {
  return src.expand(sizes, implicit);
}

at::Tensor ImmutRepeat(at::Tensor self, at::IntArrayRef sizes) {
  return self.repeat(sizes);
}

at::Tensor ImmutRepeatRev(
    at::Tensor self,
    at::Tensor src,
    at::IntArrayRef sizes) {
  auto immut_self = self;
  immut_self.repeat(sizes).copy_(src);
  return immut_self;
}

at::Tensor ImmutExpandAs(at::Tensor src, at::Tensor other) {
  return src.expand_as(other);
}

at::Tensor ImmutExpandAsRev(at::Tensor self, at::Tensor src, at::Tensor other) {
  auto immut_self = self;
  immut_self.expand_as(other).copy_(src);
  return immut_self;
}

at::Tensor ImmutIndex_Tensor(
    at::Tensor src,
    c10::List<c10::optional<at::Tensor>> indices) {
  return src.index(indices);
}

static auto _registry =
    RegisterOperators()
        .op("immut::access(Tensor src) -> Tensor",
            Access,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor",
            Assign,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::select(Tensor src, int dim, int index) -> Tensor",
            ImmutSelect,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::select_rev(Tensor self, Tensor src, int dim, int index) "
            "-> Tensor",
            ImmutSelectRev,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
            "end=None, SymInt step=1) -> Tensor",
            ImmutSlice,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::slice_rev(Tensor self, Tensor src, int dim=0, SymInt? "
            "start=None, SymInt? end=None, SymInt step=1) -> Tensor",
            ImmutSliceRev,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::squeeze(Tensor self, int dim) -> Tensor",
            ImmutSqueeze,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::unsqueeze(Tensor self, int dim) -> Tensor",
            ImmutUnqueeze,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::view(Tensor self, int[] size) -> Tensor",
            ImmutView,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::reshape(Tensor self, int[] size) -> Tensor",
            ImmutReshape,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::permute(Tensor self, int[] sizes) -> Tensor",
            ImmutPermute,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::permute_rev(Tensor self, int[] size) -> Tensor",
            ImmutPermuteRev,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
            ImmutExpand,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::expand_rev(Tensor self, int[] size, *, bool implicit) -> Tensor",
            ImmutExpandRev,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::repeat(Tensor self, int[] size) -> Tensor",
            ImmutRepeat,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::repeat_rev(Tensor self, Tensor src, int[] size) -> Tensor",
            ImmutRepeatRev,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::expand_as(Tensor self, Tensor other) -> Tensor",
            ImmutExpandAs,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::expand_as_rev(Tensor self, Tensor src, Tensor other) -> Tensor",
            ImmutExpandAsRev,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
            ImmutIndex_Tensor,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION));
static auto x = 1;
} // namespace jit
} // namespace torch
