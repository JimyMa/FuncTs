#include "tensor_ssa.h"

#include "common_passes.h"
#include "parallelize_loops.h"
#include "util/disjoint_set.h"
#include "util/ir.h"
#include "util/traits.h"

namespace torch {
namespace jit {

static bool _registered = false;

RegisterOperators registerTssaOps() {
  if (_registered) return {};
  _registered = true;
  return RegisterOperators()
      .op("tssa::Assign(Tensor self, Tensor src) -> Tensor")
      .op("tssa::Update(Tensor self, Tensor cause) -> Tensor");
}

static Node *rewriteMutating(
    Node *node, Graph *graph, DisjointSets<Value *> &aliasSets,
    std::vector<Value *> &mutValues,
    std::unordered_map<Value *, std::vector<Node *>> &mutNodes) {
  // Replace mutating operations with non-mutating ones
  auto block = node->owningBlock();
  auto mutated = node->input(0);
  Node *assignNode = nullptr;
  switch (node->kind()) {
    case aten::copy_: {
      assignNode =
          createTssaAssign(graph, mutated, node->input(1))->copyMetadata(node);
      replace(node, assignNode);
      break;
    }

    case aten::index_put_: {
      // Create imaginary advanced indexing view
      auto indexNode =
          graph->create(aten::index, {node->input(0), node->input(1)})
              ->copyMetadata(node);
      indexNode->insertAfter(node);
      aliasSets.merge(indexNode->input(0), indexNode->output(0));

      // Create assignment to the imaginary view
      mutated = indexNode->output(0);
      assignNode = createTssaAssign(graph, mutated, node->input(2));
      assignNode->insertAfter(indexNode);
      TORCH_CHECK(!node->hasUses());
      node->destroy();
      break;
    }

    default: {
      // Create immutable operation node
      auto mutSym = node->kind();
      std::string immutOpName(mutSym.toUnqualString());
      immutOpName.pop_back();
      auto immutSym = Symbol::fromQualString(
          std::string(mutSym.ns().toUnqualString()) + "::" + immutOpName);
      auto opNode = graph->create(immutSym, node->inputs())->copyMetadata(node);
      opNode->insertBefore(node);
      TORCH_CHECK(opNode->maybeSchema());

      // Create assignment node
      assignNode = createTssaAssign(graph, mutated, opNode->output(0));
      replace(node, assignNode);
      break;
    }
  }
  TORCH_CHECK(assignNode);

  // Update aliases of the assigned value
  auto afterAssign = assignNode->output(0);
  auto lastNode = assignNode;
  for (auto alias : aliasSets.getSetOf(mutated)) {
    auto mutNode = assignNode;
    if (alias != mutated) {
      auto updateNode = createTssaUpdate(graph, alias, afterAssign);
      updateNode->insertAfter(lastNode);
      mutNode = updateNode;
      lastNode = updateNode;
    }
    // Add to mutation record
    if (mutNodes.count(alias))
      mutNodes[alias].push_back(mutNode);
    else {
      mutValues.push_back(alias);
      mutNodes[alias] = {mutNode};
    }
  }

  return lastNode;
}

static void addMutatedValueToBlock(
    Value *mutated, Block *block, std::unordered_set<Block *> &visitedBlocks,
    std::unordered_map<Value *, Value *> &valueToMut, bool handleNode = true) {
  // Skip if this block if visited before
  if (visitedBlocks.count(block)) return;
  visitedBlocks.insert(block);

  // Add to block and node returns
  block->insertOutput(block->outputs().size(), mutated);
  auto node = block->owningNode();
  if (handleNode) {
    auto nodeRet = node->addOutput();
    valueToMut.insert({nodeRet, mutated});
  }

  // Handle values that are specific to node kinds
  switch (node->kind()) {
    case prim::Loop: {
      // add to block parameter of loop body
      auto param = block->addInput();
      valueToMut.insert({param, mutated});
      // add to argument of loop node
      node->addInput(mutated);
      break;
    }

    case prim::If: {
      // add to the block of the other branch
      auto blockId = block == node->blocks()[1];
      addMutatedValueToBlock(mutated, node->blocks()[!blockId], visitedBlocks,
                             valueToMut, false);
      break;
    }
  }
}

static void renameValues(
    Block *block, std::unordered_map<Value *, Value *> &valueToMut,
    std::unordered_map<Value *, std::vector<Value *>> &renameStacks) {
  // Initialize rename counts in current scope
  std::unordered_map<Value *, size_t> renameCounts;
  auto updateValue = [&](Value *value) {
    // find mutated version of this value
    Value *mutated = nullptr;
    if (valueToMut.count(value))
      mutated = valueToMut[value];
    else {
      auto defNode = value->node();
      auto kind = defNode->kind();
      if (kind == tssa::Assign || kind == tssa::Update) {
        mutated = valueToMut[defNode->input(0)];
        valueToMut.insert({value, mutated});
      }
    }
    if (!mutated) return;
    // add to rename stack
    renameStacks[mutated].push_back(value);
    // add to rename counts
    if (renameCounts.count(mutated))
      renameCounts[mutated]++;
    else
      renameCounts.insert({mutated, 1});
  };
  auto replaceInputsOf = [&](Node *node) {
    for (auto i = 0u; i < node->inputs().size(); i++) {
      auto input = node->input(i);
      if (!valueToMut.count(input)) continue;
      auto mutated = valueToMut[input];
      auto latest = renameStacks[mutated].back();
      node->replaceInput(i, latest);
    }
  };

  // Add parameters to rename stack
  for (auto param : block->inputs()) updateValue(param);

  // Process each node
  for (auto node : block->nodes()) {
    // replace inputs
    replaceInputsOf(node);
    // visit owned blocks
    for (auto nested : node->blocks())
      renameValues(nested, valueToMut, renameStacks);
    // update outputs
    for (auto output : node->outputs()) updateValue(output);
  }

  // Process return node
  replaceInputsOf(block->return_node());

  // Restore rename stack
  for (auto &pair : renameCounts) {
    for (auto i = 0u; i < pair.second; i++) renameStacks[pair.first].pop_back();
  }
}

static void removeDeadUpdateInLoop(Node *loop) {
  auto block = loop->blocks().front();
  for (auto i = 0; i < loop->outputs().size(); i++) {
    // Check if the value is dead
    auto loopOut = loop->output(i);
    if (loopOut->hasUses()) continue;
    auto blockRet = block->outputs()[i + 1];
    auto update = blockRet->node();
    if (update->kind() != tssa::Update) continue;
    if (update->input(0) != block->inputs()[i + 1]) continue;

    // Erase dead update
    loop->eraseOutput(i);
    block->eraseOutput(i + 1);
    update->destroy();
    block->eraseInput(i + 1);
    loop->removeInput(i + 2);
    i--;
  }
}

void ToTensorSSA(const std::shared_ptr<Graph> &graph) {
  // Find all mutated tensors and remove mutation
  DisjointSets<Value *> aliasSets;
  std::vector<Value *> mutValues;
  std::unordered_map<Value *, std::vector<Node *>> mutNodes;
  rewrite(graph->block(), [&](Node *node) -> Node * {
    // Skip non-tensor operations
    if (node->inputs().empty() || node->outputs().empty()) return nullptr;
    if (node->input(0)->type()->kind() != TypeKind::TensorType ||
        node->output(0)->type()->kind() != TypeKind::TensorType)
      return nullptr;

    // Rewrite mutating nodes to remove mutation
    if (isMutating(node)) {
      return rewriteMutating(node, graph.get(), aliasSets, mutValues, mutNodes);
    }

    // Extend tensor alias graph if the node is aliasing
    if (isAliasing(node)) aliasSets.merge(node->input(0), node->output(0));

    return nullptr;
  });

  // Add block parameters and returns for out-of-block mutation
  std::unordered_map<Value *, Value *> valueToMut;
  for (auto mutated : mutValues) {
    valueToMut.insert({mutated, mutated});
    auto defBlock = mutated->node()->owningBlock();
    std::unordered_set<Block *> visitedBlocks;
    auto &nodes = mutNodes[mutated];
    for (auto node : nodes) {
      for (auto block = node->owningBlock(); block != defBlock;
           block = block->owningNode()->owningBlock()) {
        addMutatedValueToBlock(mutated, block, visitedBlocks, valueToMut);
      }
    }
  }

  // Replace placeholders with real SSA values
  std::unordered_map<Value *, std::vector<Value *>> renameStacks;
  for (auto value : mutValues) renameStacks.insert({value, {}});
  renameValues(graph->block(), valueToMut, renameStacks);

  // Eliminate redundant updates
  EliminateDeadCodeTSSA(graph);
  traversePostOrder(graph->block(), [](Node *loop) {
    if (loop->kind() == prim::Loop) removeDeadUpdateInLoop(loop);
    return true;
  });

  // Eliminate redundant assignment
  rewrite(graph->block(), [](Node *node) -> Node * {
    if (node->kind() != tssa::Assign) return nullptr;
    auto output = node->output(0);
    for (auto &use : output->uses()) {
      if (use.user->kind() == tssa::Update) return nullptr;
    }
    output->replaceAllUsesWith(node->input(1));
    return remove(node);
  });
}

static void toMutableTensorsIn(Block *block) {
  auto graph = block->owningGraph();
  for (auto node = block->nodes().front(); node != block->nodes().back();
       node = node->next()) {
    auto kind = node->kind();
    if (kind == tssa::Assign) {
      auto dst = node->input(0), src = node->input(1);
      auto dstDef = dst->node();
      graph->setInsertPoint(node->next());
      if (dstDef->kind() == aten::index) {  // advanced indexing
        auto self = dstDef->input(0), indices = dstDef->input(1);
        graph->insert(aten::index_put_, {self, indices, src});
        node->output(0)->replaceAllUsesWith(self);
      } else {
        graph->insert(aten::copy_, {dst, src});
        node->output(0)->replaceAllUsesWith(dst);
      }
    } else if (kind == tssa::Update) {
      node->output(0)->replaceAllUsesWith(node->input(0));
    }
    if (kind != prim::FusionGroup) {
      for (auto subBlock : node->blocks()) toMutableTensorsIn(subBlock);
    }
  }
}

void ToMutableTensors(const std::shared_ptr<Graph> &graph) {
  toMutableTensorsIn(graph->block());
  EliminateDeadCodeTSSA(graph);
}

}  // namespace jit
}  // namespace torch
