#include "validate_graph.h"

namespace torch {
namespace jit {

static void validate(Block *block, std::unordered_set<Value *> &defined) {
  // Initialize local stack
  std::vector<Value *> stack;
  auto checkDefined = [&](Value *value, Node *node) {
    if (defined.count(value)) return;
    std::stringstream ss;
    ss << "%" + value->debugName() + " used without defined before: \n"
       << *node;
    throw c10::Error(ss.str(), "");
  };
  auto pushValue = [&](Value *value) {
    defined.insert(value);
    stack.push_back(value);
  };

  // Traverse the block
  for (auto param : block->inputs()) pushValue(param);
  for (auto node : block->nodes()) {
    if (node->hasAttribute(attr::Subgraph)) Validate(node->g(attr::Subgraph));
    for (auto input : node->inputs()) checkDefined(input, node);
    for (auto subBlock : node->blocks()) validate(subBlock, defined);
    for (auto output : node->outputs()) pushValue(output);
  }
  for (auto ret : block->outputs()) checkDefined(ret, block->return_node());

  // Pop values
  for (auto value : stack) defined.erase(value);
}

void Validate(const std::shared_ptr<Graph> &graph) {
  std::unordered_set<Value *> defined;
  validate(graph->block(), defined);
}

}  // namespace jit
}  // namespace torch
