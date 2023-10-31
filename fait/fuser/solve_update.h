#ifndef LONG_TAIL_SOLVE_UPDATE_H
#define LONG_TAIL_SOLVE_UPDATE_H

#include <memory>

#include "torch/csrc/jit/ir/ir.h"

namespace torch {
namespace jit {

struct BufferNode;
using BufferNodePtr = std::shared_ptr<BufferNode>;

struct BufferNode {
 public:
  BufferNode(Value* value) : value_(value) {}

  Value* value() { return value_; }

  std::vector<BufferNodePtr>& views() { return views_; }

  void add_view(BufferNodePtr buffer_node) { views_.push_back(buffer_node); }

 private:
  Value* value_;
  std::vector<BufferNodePtr> views_;
};

struct BufferTree;
using BufferTreePtr = std::shared_ptr<BufferTree>;

struct BufferTree {
 public:
  BufferTree(BufferNodePtr node) : root_(node) {}

  BufferNodePtr root() { return root_; }

  std::vector<BufferNodePtr> pre_order() {
    std::vector<BufferNodePtr> pre_order_list;
    std::function<void(BufferNodePtr)> visit;
    visit = [&](BufferNodePtr node) {
      pre_order_list.push_back(node);
      auto views = node->views();
      for (auto view : views) {
        visit(view);
      }
    };
    visit(root_);
    return pre_order_list;
  }

  BufferNodePtr find_buffer_node(Value* value) {
    auto pre_order_list = pre_order();
    for (auto node : pre_order_list)
      if (node->value() == value) return node;
    return nullptr;
  }

  BufferNodePtr find_parent(BufferNodePtr child) {
    auto pre_order_list = pre_order();
    for (auto node : pre_order_list) {
      auto views = node->views();
      std::vector<BufferNodePtr>::iterator iter =
          find(views.begin(), views.end(), child);
      if (iter != views.end()) {
        return node;
      }
    }
    return nullptr;
  }

 private:
  BufferNodePtr root_;
};

// struct BufferDisjointSet;

class BufferDisjointSet {
 public:
  BufferDisjointSet() {}

  std::vector<Node*> get_pass_down_chain(Value* value);
  std::vector<Node*> get_pass_up_chain(Value* value);

  BufferTreePtr find_buffer_tree(Value* value) {
    for (auto& root : roots_) {
      if (root->find_buffer_node(value)) return root;
    }
    return nullptr;
  }

  void add_value(Value* parent, Value* value) {
    auto buffer_node = std::make_shared<BufferNode>(value);
    if (auto root = find_buffer_tree(parent)) {
      root->find_buffer_node(parent)->add_view(buffer_node);
    } else {
      auto parent_buffer_node = std::make_shared<BufferNode>(parent);
      roots_.push_back(std::make_shared<BufferTree>(parent_buffer_node));
      parent_buffer_node->add_view(buffer_node);
    }
  }

 private:
  std::vector<BufferTreePtr> roots_{};
};

using BufferDisjointSetPtr = std::shared_ptr<BufferDisjointSet>;

class UpdateSolver {
 public:
  UpdateSolver(const std::shared_ptr<Graph>& graph);
  void run();

 private:
  BufferDisjointSetPtr buffer_disjoint_set_;
  std::shared_ptr<Graph> graph_;
};

void SolveUpdate(const std::shared_ptr<Graph>& graph);
}  // namespace jit
}  // namespace torch

#endif  // LONG_TAIL_SOLVE_UPDATE_H
