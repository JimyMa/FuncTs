#include "canonicalize.h"

#include "type_utils.h"
#include "util/ir.h"

namespace torch {
namespace jit {

#define REWRITE_PARAMS Node *node, Graph *graph

Node *rewriteArange(REWRITE_PARAMS) {
  // Add `start` if it is not provided
  auto zero = graph->insertConstant(int64_t(0));
  auto inputs = node->inputs().vec();
  inputs.insert(inputs.begin(), zero);
  auto newNode = graph->create(aten::arange, inputs)->copyMetadata(node);
  TORCH_CHECK(newNode->maybeOperator());
  node->output(0)->replaceAllUsesWith(newNode->output(0));
  return replace(node, newNode);
}

Node *rewriteMaxOther(REWRITE_PARAMS) {
  node->replaceWithNewSymbol(aten::maximum);
  return nullptr;
}

Node *rewriteMinOther(REWRITE_PARAMS) {
  node->replaceWithNewSymbol(aten::minimum);
  return nullptr;
}

Node *rewriteNew(REWRITE_PARAMS) {
  // Convert `aten::new_xxxs` to `aten::xxxs`
  auto self = node->input(0), size = node->input(1), dtype = node->input(2),
       device = node->input(4);
  if (dtype->type()->kind() == TypeKind::NoneType)
    dtype = graph->insert(prim::dtype, {self});
  if (device->type()->kind() == TypeKind::NoneType)
    device = graph->insert(prim::device, {self});
  auto symbol =
      Symbol::aten(std::string(node->kind().toUnqualString()).substr(4));
  auto newOut =
      graph->insert(symbol, {size}, {{"dtype", dtype}, {"device", device}},
                    node->sourceRange());
  node->output(0)->replaceAllUsesWith(newOut);
  return remove(node);
}

Node *rewritePowTensorScalar(REWRITE_PARAMS) {
  // Replace with a series of multiplication if exponent is constant
  auto self = node->input(0);
  auto exp = constant_as<int64_t>(node->input(1));
  if (!exp) return nullptr;
  Value *base = nullptr;
  while (*exp > 0) {
    if ((*exp & 1) && base) self = graph->insert(aten::mul, {self, base});
    if (base)
      base = graph->insert(aten::mul, {base, base});
    else
      base = self;
    *exp /= 2;
  }
  self->node()->copyMetadata(node);
  node->output(0)->replaceAllUsesWith(self);
  return remove(node);
}

Node *rewriteSlice(REWRITE_PARAMS) {
  // Remove if both start and end are not specified
  auto start = toIValue(node->input(2)), end = toIValue(node->input(3));
  if (start && start->isNone() && end && end->isNone()) {
    node->output(0)->replaceAllUsesWith(node->input(0));
    return remove(node);
  }
  return nullptr;
}

Node *rewriteT(REWRITE_PARAMS) {
  auto self = node->input(0);
  auto zero = graph->insertConstant(int64_t(0));
  auto one = graph->insertConstant(int64_t(1));
  auto newOut = graph->insert(aten::transpose, {self, zero, one}, {},
                              node->sourceRange());
  node->output(0)->replaceAllUsesWith(newOut);
  return remove(node);
}

Node *rewriteToDtype(REWRITE_PARAMS) {
  auto self = node->input(0), dtype = node->input(1);
  auto device = graph->insert(prim::device, {self}, {});
  auto newOut =
      graph->insert(aten::to, {self, device, dtype}, {}, node->sourceRange());
  node->output(0)->replaceAllUsesWith(newOut);
  return remove(node);
}

Node *rewriteToOther(REWRITE_PARAMS) {
  // Explicitly define dtype and device
  auto self = node->input(0), other = node->input(1);
  auto device = graph->insert(prim::device, {other}, {});
  auto dtype = graph->insert(prim::dtype, {other}, {});
  auto newOut =
      graph->insert(aten::to, {self, device, dtype}, {}, node->sourceRange());
  node->output(0)->replaceAllUsesWith(newOut);
  return remove(node);
}

Node *rewriteView(REWRITE_PARAMS) {
  node->replaceWithNewSymbol(aten::reshape);
  return nullptr;
}

OperatorMap<Node *(*)(REWRITE_PARAMS)> rewriteFuncs{
    {"aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, "
     "Device? device=None, bool? pin_memory=None) -> Tensor",
     rewriteArange},
    {"aten::max.other(Tensor self, Tensor other) -> Tensor", rewriteMaxOther},
    {"aten::min.other(Tensor self, Tensor other) -> Tensor", rewriteMinOther},
    {"aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, "
     "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
     "Tensor",
     rewriteNew},
    {"aten::new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, "
     "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
     "Tensor",
     rewriteNew},
    {"aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
     rewritePowTensorScalar},
    {"aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor(a)",
     rewriteSlice},
    {"aten::t(Tensor(a) self) -> Tensor(a)", rewriteT},
    {"aten::to.dtype(Tensor(a) self, ScalarType dtype, bool "
     "non_blocking=False, "
     "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
     rewriteToDtype},
    {"aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, "
     "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
     rewriteToOther},
    {"aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)", rewriteView},
};

void CanonicalizeOps(const std::shared_ptr<Graph> &graph) {
  rewrite(graph->block(), [&](Node *node) -> Node * {
    if (node->isMemberOf(rewriteFuncs)) {
      auto func = rewriteFuncs.find(node->getOperator());
      graph->setInsertPoint(node);
      return (*func)(node, graph.get());
    } else
      return nullptr;
  });
}

}  // namespace jit
}  // namespace torch
