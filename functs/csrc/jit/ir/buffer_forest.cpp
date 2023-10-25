#include <algorithm>
#include <cstddef>
#include <functs/csrc/jit/ir/buffer_forest.h>
#include <memory>

namespace torch {
namespace jit {

bool BufferTree::find(Value *v) {
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

void BufferTree::addEdgeToBufferTree(Value *from, Value *to) {
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
    } else {
      for (auto &child : node->pointedFrom)
        visit(child);
    }
  };
  visit(root);
}

std::shared_ptr<BufferTree> BufferForest::getBufferTreeOrNone(Value *v) {
  for (auto &bufferTree : bufferForest_) {
    if (bufferTree->find(v))
      return bufferTree;
  }
  return nullptr;
}

void BufferForest::addEdgeToBufferForest(Value *from, Value *to) {
  if (auto bufferTree = getBufferTreeOrNone(from)) {
    bufferTree->addEdgeToBufferTree(from, to);
  } else {
    std::shared_ptr<BufferNode> buffer_node_from =
        std::make_shared<BufferNode>(from);
    std::shared_ptr<BufferNode> buffer_node_to =
        std::make_shared<BufferNode>(to);
    buffer_node_from->pointsTo = buffer_node_to;
    buffer_node_to->pointedFrom.push_back(buffer_node_from);

    bufferTree = std::make_shared<BufferTree>(buffer_node_to);
    bufferForest_.push_back(bufferTree);
  }
}

void BufferForest::addMutationToBufferForest(Node *node) {
  auto tree = getBufferTreeOrNone(node->output());
  if (!tree) {
    return;
  }
  auto bufferTree = getBufferTreeOrNone(node->output());
  bufferTree->mutations.push_back(node);
}
bool BufferForest::isBufferMutation(Node *node) {
  auto tree = getBufferTreeOrNone(node->output());
  if (!tree) {
    return false;
  }
  return std::find(tree->mutations.begin(), tree->mutations.end(), node) !=
         tree->mutations.end();
}

} // namespace jit
} // namespace torch
