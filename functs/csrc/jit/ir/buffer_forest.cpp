#include <c10/util/Exception.h>
#include <functs/csrc/jit/ir/buffer_forest.h>
#include <algorithm>
#include <cstddef>
#include <deque>
#include <iostream>
#include <memory>

namespace torch {
namespace jit {

bool BufferTree::find(Value* v) {
  std::function<bool(std::shared_ptr<BufferNode>)> visit;
  visit = [&](std::shared_ptr<BufferNode> node) -> bool {
    if (node->bufferNode_->var == v)
      return true;
    else {
      bool find = false;
      for (auto child : node->pointedFrom) {
        find |= visit(child);
      }
      return find;
    }
  };
  return visit(root);
}

std::shared_ptr<BufferNode> BufferTree::getBufferNodeOrNone(Value* v) {
  struct TraversalNode {
    TraversalNode(std::shared_ptr<BufferNode> n) : node(n), visited(false) {}
    std::shared_ptr<BufferNode> node;
    bool visited;
  };

  std::deque<std::shared_ptr<TraversalNode>> stack;
  stack.push_back(std::make_shared<TraversalNode>(root));

  while (!stack.empty()) {
    auto traversalNode = stack.back();
    stack.pop_back();
    if (traversalNode->visited) {
      if (traversalNode->node->bufferNode_->var == v)
        return traversalNode->node;
    } else {
      traversalNode->visited = true;
      stack.push_back(traversalNode);
      for (auto& child : traversalNode->node->pointedFrom)
        stack.push_back(std::make_shared<TraversalNode>(child));
    }
  }

  // std::cout << "\033[1;31;40m[WARN]\033[0m  Cannot find node for %"
  //           << v->debugName() << ", forget it when buffer forest
  //           construction"
  //           << std::endl;

  return nullptr;
}

void BufferTree::addEdgeToBufferTree(Value* from, Value* to) {
  if (from == root->bufferNode_->var) {
    auto to_node = std::make_shared<BufferNode>(to);
    to_node->pointedFrom.push_back(root);
    root->pointsTo = to_node;
    root = to_node;
    return;
  }
  std::function<void(std::shared_ptr<BufferNode>)> visit;
  visit = [&](std::shared_ptr<BufferNode> node) {
    if (to == node->bufferNode_->var) {
      auto from_node = std::make_shared<BufferNode>(from);
      node->pointedFrom.push_back(from_node);
      from_node->pointsTo = node;
    } else {
      for (auto& child : node->pointedFrom)
        visit(child);
    }
  };
  visit(root);
}

void BufferTree::dump() const {
  struct TraversalNode {
    TraversalNode(std::shared_ptr<BufferNode> n) : node(n), visited(false) {}
    bool visited;
    std::shared_ptr<BufferNode> node;
  };

  std::cout << "root value: %" << root->bufferNode_->var->debugName()
            << std::endl;
  std::cout << "=== 1. Points To Relationship ===" << std::endl;

  // echo a chain from root node to leaf node
  std::deque<std::shared_ptr<TraversalNode>> stack;
  stack.push_back(std::make_shared<TraversalNode>(root));
  while (!stack.empty()) {
    auto traversalNode = stack.back();
    stack.pop_back();
    if (traversalNode->visited) {
      if (traversalNode->node->pointedFrom.empty()) {
        auto wanderingNode = traversalNode->node;
        std::cout << "%" << wanderingNode->bufferNode_->var->debugName();
        while (wanderingNode->pointsTo) {
          std::cout << " => "
                    << wanderingNode->pointsTo->bufferNode_->var->debugName();
          wanderingNode = wanderingNode->pointsTo;
        }
        std::cout << std::endl;
      }
    } else {
      traversalNode->visited = true;
      stack.push_back(traversalNode);
      for (auto& point_from : traversalNode->node->pointedFrom) {
        stack.push_back(std::make_shared<TraversalNode>(point_from));
      }
    }
  }
  std::cout << std::endl;
  std::cout << "=== 2. Mutated Node ===" << std::endl;
  for (auto& node : mutations)
    node->dump();
}

std::shared_ptr<BufferTree> BufferForest::getBufferTreeOrNone(Value* v) {
  for (auto& bufferTree : bufferForest_) {
    if (bufferTree->find(v))
      return bufferTree;
  }
  // std::cout << "\033[1;31;40m[WARN]\033[0m  Cannot find tree for %"
  //           << v->debugName() << ", forget it when buffer forest
  //           construction"
  //           << std::endl;

  return nullptr;
}

std::shared_ptr<BufferNode> BufferForest::getBufferNodeOrNone(Value* v) {
  if (auto tree = getBufferTreeOrNone(v))
    return tree->getBufferNodeOrNone(v);
  return nullptr;
}

void BufferForest::mergeBufferTree(Value* from, Value* to) {
  auto from_tree = getBufferTreeOrNone(from);
  auto from_node = from_tree->getBufferNodeOrNone(from);
  auto to_tree = getBufferTreeOrNone(to);
  auto to_node = to_tree->getBufferNodeOrNone(to);

  AT_ASSERT(
      from_tree && to_tree,
      "If merge two trees, from_tree and to_tree must both exist!!!");

  AT_ASSERT(
      from_node && to_node,
      "If merge two trees, from_node and to_node must both exist!!!");

  from_node->pointsTo = to_node;
  to_node->pointedFrom.push_back(from_node);

  bufferForest_.erase(from_tree);
}

void BufferForest::replaceValue(Value* from, Value* to) {
  auto from_node = getBufferTreeOrNone(from)->getBufferNodeOrNone(from);
  from_node->bufferNode_->var = to;
}

void BufferForest::replaceMutation(Node* from, Node* to) {
  for (auto& bufferTree : bufferForest_) {
    if (bufferTree->mutations.count(from)) {
      bufferTree->mutations.insert(to);
      bufferTree->mutations.erase(from);
      return;
    }
  }
  // std::cout << "\033[1;31;40m[WARN]\033[0m cannot find mutation when replace
  // "
  //              "mutation!!!"
  //           << std::endl;
  return;
}

void BufferForest::addEdgeToBufferForest(Value* from, Value* to) {
  auto from_tree = getBufferTreeOrNone(from);
  auto to_tree = getBufferTreeOrNone(to);

  std::set<int> f;

  if (from_tree && to_tree) {
    mergeBufferTree(from, to);
  } else if (from_tree) {
    auto bufferTree = getBufferTreeOrNone(from);
    from_tree->addEdgeToBufferTree(from, to);
  } else if (to_tree) {
    auto bufferTree = getBufferTreeOrNone(to);
    to_tree->addEdgeToBufferTree(from, to);
  } else {
    std::shared_ptr<BufferNode> buffer_node_from =
        std::make_shared<BufferNode>(from);
    std::shared_ptr<BufferNode> buffer_node_to =
        std::make_shared<BufferNode>(to);
    buffer_node_from->pointsTo = buffer_node_to;
    buffer_node_to->pointedFrom.push_back(buffer_node_from);

    auto bufferTree = std::make_shared<BufferTree>(buffer_node_to);
    bufferForest_.insert(bufferTree);
  }
}

void BufferForest::addMutationToBufferForest(Node* node) {
  auto tree = getBufferTreeOrNone(node->output());
  if (!tree) {
    return;
  }
  auto bufferTree = getBufferTreeOrNone(node->output());
  bufferTree->mutations.insert(node);
}

bool BufferForest::isBufferMutation(Node* node) {
  for (auto& bufferTree : bufferForest_) {
    if (std::find(
            bufferTree->mutations.begin(), bufferTree->mutations.end(), node) !=
        bufferTree->mutations.end())
      return true;
  }
  return false;
}

void BufferForest::dump() const {
  // dump buffer forest for each buffer tree
  int cnt = 0;
  for (auto& tree : bufferForest_) {
    std::cout << "*** buffer " << cnt++ << " ***" << std::endl;
    tree->dump();
    std::cout << "*** **** **** ***" << std::endl << std::endl;
  }
}

} // namespace jit
} // namespace torch
