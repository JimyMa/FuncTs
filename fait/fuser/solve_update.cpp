#include "fuser/solve_update.h"

#include <ATen/core/interned_strings.h>
#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <iostream>
#include <unordered_set>

#include "passes/tensor_ssa.h"
#include "tssa_set_ops.h"

namespace torch {
namespace jit {

std::unordered_set<Symbol> view_symbol = {
    aten::select,
    aten::slice,
    tssa::Assign,
};

std::unordered_map<Symbol, Symbol> view_reverse_map = {
    {aten::select, tssa::SelectSet},
    {aten::slice, tssa::SliceSet}};

std::vector<Node*> BufferDisjointSet::get_pass_up_chain(Value* value) {
  std::vector<Node*> result;
  auto buffer_tree = find_buffer_tree(value);
  if (!buffer_tree)
    std::cout << "cannot find buffer tree when solve update "
              << value->debugName() << std::endl;
  AT_ASSERT(buffer_tree, "cannot find buffer tree when solve update");
  auto buffer_node = buffer_tree->find_buffer_node(value);
  result.push_back(buffer_node->value()->node());
  while ((buffer_node = buffer_tree->find_parent(buffer_node)))
    result.push_back(buffer_node->value()->node());
  return result;
}

std::vector<Node*> BufferDisjointSet::get_pass_down_chain(Value* value) {
  std::vector<Node*> result;
  result = get_pass_up_chain(value);
  std::reverse(result.begin(), result.end());
  return result;
}

UpdateSolver::UpdateSolver(const std::shared_ptr<Graph>& graph)
    : buffer_disjoint_set_(std::make_shared<BufferDisjointSet>()),
      graph_(graph) {}

void UpdateSolver::run() {
  auto block = graph_->block();
  for (auto node = block->nodes().front(); node != block->nodes().back();) {
    if (view_symbol.count(node->kind())) {
      buffer_disjoint_set_->add_value(node->input(0), node->output(0));
    } else if (node->kind() == c10::tssa::Update) {
      auto src = node->input(0);
      auto updator = node->input(1);
      auto updated = node->output(0);
      std::vector<Node*> pass_up_chain =
          buffer_disjoint_set_->get_pass_up_chain(updator);
      std::vector<Node*> pass_down_chain =
          buffer_disjoint_set_->get_pass_down_chain(src);

      Value* clue_value = src;
      Node* pre_node = node;
      Node* pre_new_node = node;

      for (int i = 0; i < pass_up_chain.size() - 1; i++) {
        auto wandering_node = pass_up_chain[i];

        if (wandering_node->kind() == c10::tssa::Assign) {
          clue_value = wandering_node->input(1);
          continue;
        }
        Symbol new_node_kind = view_reverse_map[wandering_node->kind()];
        Node* new_node = graph_->create(new_node_kind, 1);
        for (auto input : wandering_node->inputs())
          new_node->addInput(input);
        new_node->insertInput(1, clue_value);
        clue_value = new_node->output(0);
        clue_value->copyMetadata(new_node->input(0));
        new_node->insertAfter(pre_new_node);
        TORCH_CHECK(new_node->maybeOperator());
        pre_node = wandering_node;
        pre_new_node = new_node;
      }
      for (int i = 1; i < pass_down_chain.size() - 1; i++) {
        auto wandering_node = pass_down_chain[i];
        Symbol new_node_kind = wandering_node->kind();
        Node* new_node = graph_->create(new_node_kind, 1);
        for (auto input : wandering_node->inputs())
          new_node->addInput(input);
        clue_value = new_node->output(0);
        clue_value->copyMetadata(wandering_node->output(0));
        new_node->insertAfter(pre_node);
        pre_node = wandering_node;
      }

      updated->replaceAllUsesWith(clue_value);
      node->destroy();
      node = pre_node;
    }
    node = node->next();
  }
}

void SolveUpdate(const std::shared_ptr<Graph>& graph) {
  UpdateSolver(graph).run();
}

} // namespace jit
} // namespace torch
