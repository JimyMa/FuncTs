#include "ir.h"

#include "passes/refine_types.h"

namespace torch {
namespace jit {

bool traversePreOrder(Block *block,
                      const std::function<bool(Node *)> &visitor) {
  for (auto node : block->nodes()) {
    if (!visitor(node)) return false;
    for (auto nested : node->blocks())
      if (!traversePreOrder(nested, visitor)) return false;
  }
  return true;
}

bool traversePostOrder(Block *block,
                       const std::function<bool(Node *)> &visitor) {
  for (auto node : block->nodes()) {
    for (auto nested : node->blocks())
      if (!traversePostOrder(nested, visitor)) return false;
    if (!visitor(node)) return false;
  }
  return true;
}

bool containsAnySymbol(Block *block,
                       const std::unordered_set<Symbol> &symbols) {
  return !traversePreOrder(
      block, [&](Node *node) { return !symbols.count(node->kind()); });
}

void rewrite(Block *block, const std::function<Node *(Node *)> &pattern) {
  for (auto node = block->nodes().front(); node != block->nodes().back();
       node = node->next()) {
    for (auto nested : node->blocks()) rewrite(nested, pattern);
    auto newNode = pattern(node);
    if (newNode) node = newNode;
  }
}

void cloneNodesTo(Node *begin, Node *end, Node *point,
                  std::unordered_map<Value *, Value *> &valueMap,
                  std::unordered_map<Value *, TypePtr> *refinedTypes) {
  TORCH_CHECK(begin->owningBlock() == end->owningBlock());
  TORCH_CHECK(begin->isBefore(end) || begin == end);
  auto graph = point->owningGraph();
  for (auto iter = graph_node_list_iterator(begin, kNextDirection);
       iter != graph_node_list_iterator(end, kNextDirection); ++iter) {
    auto node = *iter;
    auto newNode = graph->createClone(
        node, [&](Value *v) { return valueMap.count(v) ? valueMap[v] : v; });
    newNode->insertBefore(point);
    for (auto i = 0u; i < node->outputs().size(); i++)
      valueMap[node->output(i)] = newNode->output(i);
    if (refinedTypes) transferRefinedTypesOf(node, newNode, *refinedTypes);
  }
}

void cloneNodesToBlock(Node *begin, Node *end, Block *block,
                       std::unordered_map<Value *, Value *> &valueMap,
                       std::unordered_map<Value *, TypePtr> *refinedTypes) {
  cloneNodesTo(begin, end, block->return_node(), valueMap, refinedTypes);
}

void moveNodesToBlock(Node *begin, Node *end, Block *block,
                      std::unordered_map<Value *, Value *> &valueMap,
                      std::unordered_map<Value *, TypePtr> *refinedTypes) {
  cloneNodesToBlock(begin, end, block, valueMap, refinedTypes);
  graph_node_list_iterator iterBegin(end->prev(), kPrevDirection),
      iterEnd(begin->prev(), kPrevDirection);
  for (auto iter = iterBegin; iter != iterEnd; ++iter) {
    TORCH_CHECK(!(*iter)->hasUses());
    iter.destroyCurrent();
  }
  if (refinedTypes) removeDeadRefinedTypes(*refinedTypes, block->owningGraph());
}

}  // namespace jit
}  // namespace torch
