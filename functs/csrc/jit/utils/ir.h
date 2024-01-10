#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/// @brief Traverse the nodes in the block in pre-order (node first, then its
/// nested blocks).
/// @param block The block to be traversed.
/// @param visitor Visitor function. Returns true if the traversal continues,
/// and aborts otherwise.
/// @return If the traversal terminates without abortion.
bool traversePreOrder(Block *block, const std::function<bool(Node *)> &visitor);

/// @brief Traverse the nodes in the block in pre-order (node first, then its
/// nested blocks).
/// @param block The block to be traversed.
/// @param visitor Visitor function. Returns true if the traversal continues,
/// and aborts otherwise.
/// @return If the traversal terminates without abortion.
bool traversePostOrder(Block *block,
                       const std::function<bool(Node *)> &visitor);

/// @brief Check if a block contains (recursively) any symbol in the symbol set.
/// @param block The block to be checked.
/// @param symbols The symbol set.
/// @return Check result.
bool containsAnySymbol(Block *block, const std::unordered_set<Symbol> &symbols);

/// @brief Remove the node, and return the one right before it.
/// @param node The node to be removed.
/// @return The node right before the removed one.
inline Node *remove(Node *node) {
  auto prev = node->prev();
  node->destroy();
  return prev;
}

/// @brief Replace the node with a new node, and return the one right before the
/// old node.
/// @param oldNode The old node to be replaced.
/// @param newNode The new node that will replace the old one.
/// @return The node right before the old one.
inline Node *replace(Node *oldNode, Node *newNode) {
  newNode->insertAfter(oldNode);
  oldNode->replaceAllUsesWith(newNode);
  return remove(oldNode);
}

/// @brief Rewrite nodes in a block with a given pattern recursively in
/// post-order. Note that the actual rewrite should be done by the user, and
/// this function will NOT mutates the IR.
/// @param block The block to be rewritten.
/// @param pattern The rewrite pattern, which returns a new node if the rewrite
/// is successfully applied and the following traversal should begin right after
/// this node, and nullptr otherwise.
void rewrite(Block *block, const std::function<Node *(Node *)> &pattern);

/// @brief Clone the nodes in range [`begin`, `end`) to the end of the new
/// block.
/// @param begin The beginning of the node range (inclusive).
/// @param end The end of the node range (exclusive). Must be in the same block
/// as `begin` and be after `begin` topologically.
/// @param point The point to clone nodes before.
/// @param valueMap Mappings from the value in the original block to the ones in
/// the new block.
/// @param refinedTypes Optional mappings of refined types for values.
void cloneNodesTo(Node *begin, Node *end, Node *point,
                  std::unordered_map<Value *, Value *> &valueMap,
                  std::unordered_map<Value *, TypePtr> *refinedTypes = nullptr);

/// @brief Clone the nodes in range [`begin`, `end`) to the end of the new
/// block.
/// @param begin The beginning of the node range (inclusive).
/// @param end The end of the node range (exclusive). Must be in the same
/// block as `begin` and be after `begin` topologically.
/// @param block The new block to clone nodes to.
/// @param valueMap Mappings from the value in the original block to the
/// ones in the new block.
/// @param refinedTypes Optional mappings of refined types for values.
void cloneNodesToBlock(
    Node *begin, Node *end, Block *block,
    std::unordered_map<Value *, Value *> &valueMap,
    std::unordered_map<Value *, TypePtr> *refinedTypes = nullptr);

/// @brief Move the nodes in range [`begin`, `end`) to the end of the new block.
/// @param begin The beginning of the node range (inclusive).
/// @param end The end of the node range (exclusive). Must be in the same block
/// as `begin` and be after `begin` topologically.
/// @param block The new block to clone nodes to.
/// @param graph The graph that owns the block.
/// @param valueMap Mappings from the value in the original block to the ones in
/// the new block.
/// @param refinedTypes Optional mappings of refined types for values.
void moveNodesToBlock(
    Node *begin, Node *end, Block *block,
    std::unordered_map<Value *, Value *> &valueMap,
    std::unordered_map<Value *, TypePtr> *refinedTypes = nullptr);

}  // namespace jit
}  // namespace torch
