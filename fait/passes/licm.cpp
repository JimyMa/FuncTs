#include "common_passes.h"
#include "parallelize_loops.h"
#include "type_utils.h"
#include "util/ir.h"
#include "util/traits.h"

namespace torch {
namespace jit {

static bool producesLoopInvariants(Node *node, Block *body) {
  // All constants are loop variants
  if (node->kind() == prim::Constant) return true;

  // Skip nodes with no outputs
  if (node->outputs().empty()) return false;

  // Skip nodes with blocks
  if (!node->blocks().empty()) return false;

  // Skip mutating nodes
  if (isMutating(node)) return false;

  // Skip tensors
  auto outputs = node->outputs();
  if (std::any_of(outputs.begin(), outputs.end(), isTensor)) return false;

  // Skip if its outputs are mutated
  if (std::any_of(outputs.begin(), outputs.end(), isMutated)) return false;

  // Check if all of its inputs are defined outside of loop body
  auto inputs = node->inputs();
  return std::all_of(inputs.begin(), inputs.end(), [&](Value *value) {
    return value->node()->owningBlock() != body;
  });
}

static void hoistInvariantsOf(Node *loop, Graph *graph) {
  auto body = loop->blocks()[0];
  for (auto iter = body->nodes().begin(); iter != body->nodes().end(); ++iter) {
    // Decide whether the node produces loop invariants
    auto node = *iter;
    if (!producesLoopInvariants(node, body)) continue;

    // Create and insert hoisted node
    auto hoistedNode = graph->createClone(
        node, [](Value *v) { return v; }, false);
    hoistedNode->insertBefore(loop);

    // Remove original node
    node->replaceAllUsesWith(hoistedNode);
    iter.destroyCurrent();
  }
}

void HoistLoopInvariants(const std::shared_ptr<Graph> &graph) {
  // Collect loops in the the graph in post-order
  std::vector<Node *> loops;
  traversePostOrder(graph->block(), [&](Node *node) {
    if (node->kind() == prim::Loop || node->kind() == prim::ParallelMap)
      loops.push_back(node);
    return true;
  });

  // Hoist loop invariants of each loop
  for (auto loop : loops) hoistInvariantsOf(loop, graph.get());
}

}  // namespace jit
}  // namespace torch
