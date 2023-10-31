#include <torch/csrc/jit/ir/node_hashing.h>

#include "common_passes.h"
#include "util/ir.h"
#include "util/traits.h"

namespace torch {
namespace jit {

using SubexprSet = std::unordered_set<Node *, HashNode, EqualNode>;

static std::unordered_set<Symbol> ignoredSymbols{prim::ListConstruct};

static bool mayConsider(Node *node) {
  // Skip nodes with no outputs
  if (node->outputs().empty()) return false;

  // Skip certain symbols
  if (ignoredSymbols.count(node->kind())) return false;

  // Skip mutating nodes
  if (isMutating(node)) return false;

  // Skip nodes with blocks
  if (!node->blocks().empty()) return false;

  // Skip nodes with mutated inputs or outputs
  for (auto input : node->inputs()) {
    if (isMutated(input)) return false;
  }
  for (auto output : node->outputs()) {
    if (isMutated(output)) return false;
  }

  return true;
}

void eliminateCommonSubexprIn(Block *block, SubexprSet &subexprs) {
  std::vector<Node *> scope;
  auto nodes = block->nodes();
  for (auto node = nodes.front(); node != nodes.back(); node = node->next()) {
    for (auto nested : node->blocks())
      eliminateCommonSubexprIn(nested, subexprs);
    if (!mayConsider(node)) continue;
    auto iter = subexprs.find(node);
    if (iter != subexprs.end()) {
      auto existing = *iter;
      node->replaceAllUsesWith(existing);
      node = remove(node);
    } else {
      subexprs.insert(node);
      scope.push_back(node);
    }
  }
  for (auto node : scope) subexprs.erase(node);
}

void EliminateCommonSubexprTSSA(const std::shared_ptr<Graph> &graph) {
  SubexprSet subexprs;
  eliminateCommonSubexprIn(graph->block(), subexprs);
}

}  // namespace jit
}  // namespace torch
