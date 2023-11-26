#include "op_stat.h"

#include <iostream>

#include "parallelize_loops.h"
#include "util/ir.h"

namespace torch {
namespace jit {

static std::unordered_set<Symbol> memOps{// Generation/Conversion
                                         aten::to,
                                         aten::contiguous,
                                         aten::zeros,
                                         aten::zeros_like,
                                         aten::ones,
                                         aten::ones_like,
                                         aten::arange,
                                         // Unary
                                         aten::exp,
                                         aten::sigmoid,
                                         aten::sigmoid_,
                                         aten::clamp,
                                         aten::clamp_,
                                         aten::triu,
                                         // Binary
                                         aten::add,
                                         aten::sub,
                                         aten::mul,
                                         aten::div,
                                         aten::__and__,
                                         aten::minimum,
                                         aten::maximum,
                                         aten::eq,
                                         aten::ne,
                                         aten::lt,
                                         aten::le,
                                         aten::gt,
                                         aten::ge,
                                         // Reduction
                                         aten::sum,
                                         aten::max,
                                         aten::min,
                                         aten::softmax,
                                         // View
                                         aten::select,
                                         aten::slice,
                                         aten::squeeze,
                                         aten::unsqueeze,
                                         aten::reshape,
                                         aten::expand,
                                         aten::expand_as,
                                         aten::permute,
                                         aten::transpose,
                                         // Copy
                                         aten::repeat,
                                         aten::cat,
                                         aten::stack,
                                         aten::index};

static std::unordered_set<Symbol> nonTensorOps{
    prim::Constant,
    prim::TupleUnpack,
    prim::ListUnpack,
    aten::__getitem__};

void CountMemoryIntensiveOps(const std::shared_ptr<Graph>& graph) {
  size_t numTensorOps = 0, numMemOps = 0;
  traversePreOrder(graph->block(), [&](Node* node) {
    if (!node->blocks().empty())
      return true;
    if (node->outputs().empty())
      return true;
    if (node->output(0)->type()->kind() != c10::TypeKind::TensorType)
      return true;
    if (nonTensorOps.count(node->kind()))
      return true;
    numTensorOps++;
    if (memOps.count(node->kind()))
      numMemOps++;
    return true;
  });
  print(std::cout, "Total operators: ", numTensorOps, '\n');
  print(std::cout, "Memory-intensive operators: ", numMemOps, '\n');
}

void CountLoops(const std::shared_ptr<Graph>& graph) {
  size_t numPMaps = 0, numLoops = 0;
  traversePreOrder(graph->block(), [&](Node* node) {
    if (node->kind() == prim::Loop)
      numLoops++;
    else if (node->kind() == prim::ParallelMap)
      numPMaps++;
    return true;
  });
  print(std::cout, "Parallelizable loops: ", numPMaps, '\n');
  print(std::cout, "Normal loops: ", numLoops, '\n');
}

} // namespace jit
} // namespace torch