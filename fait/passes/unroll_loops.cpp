#include "unroll_loops.h"

#include <torch/csrc/jit/ir/ir_views.h>

#include "parallelize_loops.h"
#include "util/ir.h"

namespace torch {
namespace jit {

static void unrollLoop(Node *loop, ValueTypeMap &refinedTypes) {
  // Prepare for unrolling
  LoopView view(loop);
  auto tripCount = *constant_as<int64_t>(view.maxTripCount());
  auto carriedValues = view.carriedInputs().vec();
  std::unordered_map<Value *, Value *> valueMap;

  // Inline loop body
  auto graph = loop->owningGraph();
  graph->setInsertPoint(loop);
  auto body = view.bodyBlock();
  for (auto i : c10::irange(tripCount)) {
    // Insert loop index
    auto idxParam = view.currentTripCount();
    auto idxVal = graph->insertConstant(i)->copyMetadata(idxParam);
    valueMap[idxParam] = idxVal;

    // Map block parameters to carried values
    TORCH_CHECK(view.bodyCarriedInputs().size() == carriedValues.size());
    for (auto i : c10::irange(view.bodyCarriedInputs().size())) {
      auto param = view.bodyCarriedInputs()[i], carried = carriedValues.at(i);
      valueMap[param] = carried;
    }

    // Clone body out of the loop
    cloneNodesTo(body->nodes().front(), body->nodes().back(), loop, valueMap,
                 &refinedTypes);

    // Retrieve carried outputs
    std::vector<Value *> newCarried;
    for (auto ret : view.bodyCarriedOutputs())
      newCarried.push_back(valueMap.at(ret));
    carriedValues.swap(newCarried);
  }

  // Replace outputs
  TORCH_CHECK(view.carriedOutputs().size() == carriedValues.size());
  for (auto i : c10::irange(view.carriedOutputs().size())) {
    auto unrollOut = carriedValues[i], loopOut = view.carriedOutputs()[i];
    loopOut->replaceAllUsesWith(unrollOut);
  }
  loop->destroy();
  removeDeadRefinedTypes(refinedTypes, graph);
}

static std::unordered_set<Symbol> forbidUnrollSymbols{
    prim::Loop, prim::If, prim::FusionGroup, prim::ParallelMap};

static bool shouldUnroll(Node *loop) {
  LoopView view(loop);
  if (view.carriedInputs().empty()) return false;
  if (view.maxTripCount()->node()->kind() != prim::Constant) return false;
  if (containsAnySymbol(view.bodyBlock(), forbidUnrollSymbols)) return false;
  return true;
}

void UnrollLoopsWithDeps(const std::shared_ptr<Graph> &graph,
                         ValueTypeMap &refinedTypes) {
  std::vector<Node *> loops;
  traversePostOrder(graph->block(), [&](Node *loop) {
    if (loop->kind() != prim::Loop) return true;
    if (shouldUnroll(loop)) loops.push_back(loop);
    return true;
  });

  for (auto loop : loops) unrollLoop(loop, refinedTypes);
}

}  // namespace jit
}  // namespace torch