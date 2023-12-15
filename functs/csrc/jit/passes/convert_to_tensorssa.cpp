#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <deque>
#include <memory>
#include <set>
#include <unordered_map>

#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <functs/csrc/jit/ir/alias_analysis.h>
#include <functs/csrc/jit/ir/buffer_forest.h>
#include <functs/csrc/jit/ir/symbol_ext.h>
#include <functs/csrc/jit/passes/convert_to_tensorssa.h>
#include <functs/csrc/jit/passes/remove_inplace.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/utils/memory_dag.h>
#include <unordered_map>
#include <vector>
// #include <torch/csrc/jit/passes/utils/memory_dag.h>

using namespace c10;

namespace torch {
namespace jit {

static void GetBufferTreeAliasDb(
    std::shared_ptr<Graph> g,
    AliasDbCopy& aliasDb_buffer_tree) {
  // g: Graph-level IR based tensor program
  // alias_db_buffer_tree: modified alias relationship

  // Step 1. visit all elements in aliasdb, delete all unsupported alias
  // relationship.
  // // 1. find all ambigious alias relationship
  // // 2. delete them from aliasDb_buffer_tree

  bool any_changed = true;
  auto elementMap = aliasDb_buffer_tree.elementMapMutable();

  ska::flat_hash_map<const Value*, Element*> ambigious_alias;

  // Step 1
  for (const auto& ptrPair : elementMap) {
    auto value = ptrPair.first;
    auto element = ptrPair.second;

    if (element->pointsTo.count() > 1 /* Step 1.1: count > 1 */ ||
        aliasDb_buffer_tree.mayAliasWildcard(
            value) /* Inter-procedure dependency */
        || prim::Loop == value->node()->kind() /* Loop carried dependency */ ||
        prim::If == value->node()->kind() /* Control flow dependency */ ||
        element->values.size() > 1 /* Container dependency */ ||
        value->type()->kind() == TypeKind::ListType ||
        value->type()->kind() == TypeKind::DictType ||
        value->type()->kind() == TypeKind::ClassType) {
      ambigious_alias[ptrPair.first] = ptrPair.second;
    }
  }

  ska::flat_hash_map<const Value*, Element*> may_alias;

  // Step 1.2
  // get may_alias
  for (const auto& ptrPair : elementMap) {
    for (const auto& ptrPairAmbigious : ambigious_alias) {
      if (aliasDb_buffer_tree.mayAlias(ptrPair.first, ptrPairAmbigious.first) &&
          !may_alias.count(ptrPair.first)) {
        may_alias.insert(ptrPair);
      }
    }
  }

  // Delete from elementMap
  for (const auto& ptrPair : may_alias) {
    aliasDb_buffer_tree.elementMapErase(ptrPair.first);
  }
}

static std::shared_ptr<BufferForest> TensorSSAGetBufferForest(
    std::shared_ptr<Graph> graph) { // AliasDb
  auto aliasDb_buffer_tree = AliasDbCopy(graph);

  GetBufferTreeAliasDb(graph, aliasDb_buffer_tree);

  auto bufferForest = std::make_shared<BufferForest>();
  auto elementMap = aliasDb_buffer_tree.elementMap();
  for (auto& elemPtr : elementMap) {
    auto value = elemPtr.first;
    auto elem = elemPtr.second;
    for (auto pointToIndex : elem->pointsTo) {
      bufferForest->addEdgeToBufferForest(
          const_cast<Value*>(value),
          const_cast<Value*>(
              *aliasDb_buffer_tree.fromIndex(pointToIndex)->values.begin()));
    }
  }
  auto writeIndex = aliasDb_buffer_tree.writeIndexMutable();
  for (auto node_vs_idx : *writeIndex) {
    bufferForest->addMutationToBufferForest(node_vs_idx.first);
  }
  return bufferForest;
}

static void TensorSSAAliasRemoval(
    Block* b,
    std::shared_ptr<BufferForest> bufferForest,
    std::shared_ptr<TensorSSAMutateInfo> mutateInfo) {
  auto nodes = b->nodes();
  for (auto node = nodes.front(); node != nodes.back();) {
    auto blocks = node->blocks();
    for (auto block : blocks) {
      TensorSSAAliasRemoval(block, bufferForest, mutateInfo);
    }

    // Judge if a node is a write node, if true, convert to immutable equivalent
    // Note: Unsupported alias has been eliminated from alias set
    if (bufferForest->isBufferMutation(node)) {
      // TODO: only tensor values are considered...
      // Step 1. Pass Up
      auto interupt = node->next();
      const Value* points_to;
      Value* leaf_value = node->output(0);
      Value* pass_up_value = node->input(0);
      Node* node_insert = node;
      do {
        // substitute to buffer node
        auto leaf_buffer_node = bufferForest->getBufferTreeOrNone(leaf_value)
                                    ->getBufferNodeOrNone(leaf_value);
        auto points_to_node = leaf_buffer_node->pointsTo;

        // go up until meet the buffer root, the origin defined tensor who owns
        // the storage.
        if (points_to_node) {
          points_to = points_to_node->bufferNode_->var;
          Node* pass_up_node;
          auto leaf_node = leaf_value->node();

          if (immutable::reverseVersion.count(leaf_node->kind())) {
            pass_up_node = b->owningGraph()->create(
                immutable::reverseVersion[leaf_node->kind()],
                leaf_value->node()->inputs(),
                1);
            if (immutable::Assign != leaf_node->kind())
              pass_up_node->insertInput(1, pass_up_value);
          } else {
            AT_ASSERT(
                false,
                "Unknown alias operator when pass up",
                leaf_node->kind().toQualString());
          }
          pass_up_node->output(0)->setType(leaf_value->type());
          pass_up_value = pass_up_node->output(0);

          pass_up_node->insertAfter(node_insert);
          node_insert = pass_up_node;
          leaf_value = const_cast<Value*>(points_to);
        } else {
          points_to = nullptr;
        }
      } while (points_to);

      // Step 2. pass down
      // Pass down non recursive implementation
      struct VisitNode {
        VisitNode(Value* root, Value* pass_down, Node* update)
            : root_value(root),
              pass_down_value(pass_down),
              update_node(update){};
        bool visted{false};
        Value* root_value;
        Value* pass_down_value;
        Node* update_node;
      };

      Value* root_value = leaf_value;
      Value* pass_down_value = pass_up_value;

      auto update_node = b->owningGraph()->create(
          tensorssa::Update,
          {pass_down_value->node()->output(), root_value},
          0);
      update_node->insertAfter(node_insert);
      mutateInfo->addMutNodes(update_node);

      auto rootVisitNode =
          std::make_shared<VisitNode>(root_value, pass_down_value, update_node);

      std::deque<std::shared_ptr<VisitNode>> stack;
      stack.push_back(rootVisitNode);

      while (!stack.empty()) {
        auto visitingNode = stack.back();
        stack.pop_back();
        if (visitingNode->visted) {
          // ACCESS Buffer Forest
          auto pass_down_buffer_node =
              bufferForest->getBufferNodeOrNone(visitingNode->root_value);
          for (auto point_from_buffer_node :
               pass_down_buffer_node->pointedFrom) {
            auto from_value = point_from_buffer_node->bufferNode_->var;
            auto from_node = from_value->node();

            // node is dominated by from_node is nesessary !!!
            // the variable `copy_` at escaping scope may cause soundness
            // concern !!! def func(a, b):
            // // if cond:
            // // // c = a[0]
            // // else:
            // // // pass
            // // c.copy_(b)
            // a tensor which is not dominated by value try to mutate the value
            // is forbidden in Functionalization NOTE: this feature is
            // unsupported in TorchScript either.

            // For easily comperahension, If from node is node, generate a new
            // immut::assign (this cond can be remove)*/

            if ((from_node->isBefore(node) && node->isDominatedBy(from_node)) ||
                from_node == node) {
              Node* pass_down_node;
              if (immutable::reverseVersion.count(from_node->kind())) {
                pass_down_node = b->owningGraph()->create(
                    from_node->kind(),
                    const_cast<Node*>(from_node)->inputs(),
                    1);
                if (immutable::Assign != from_node->kind())
                  pass_down_node->replaceInput(
                      0, visitingNode->pass_down_value);
                else {
                  pass_down_node->replaceInput(
                      0, visitingNode->pass_down_value);
                  pass_down_node->replaceInput(
                      1, visitingNode->pass_down_value);
                }
              } else {
                AT_ASSERT(
                    false,
                    "Unknown alias operator when pass down",
                    from_node->kind().toQualString());
              }
              // b->owningGraph()->insertNode(pass_down_node);
              pass_down_node->insertAfter(
                  visitingNode->pass_down_value->node());

              pass_down_value = pass_down_node->output(0);
              pass_down_value->copyMetadata(from_node->output(0));

              // generate a strong update to beacon mutation
              auto update_node = b->owningGraph()->create(
                  tensorssa::Update, {pass_down_node->output(), from_value}, 0);
              update_node->insertAfter(visitingNode->update_node);

              mutateInfo->addMutNodes(update_node);

              root_value = from_node->output(0);
              auto leafVisitingNode = std::make_shared<VisitNode>(
                  root_value, pass_down_value, update_node);
              stack.push_back(leafVisitingNode);
            }
          }
        } else {
          visitingNode->visted = true;
          stack.push_back(visitingNode);
        }
      }
      node = interupt;
    } else {
      node = node->next();
    }
  }
}

void TensorSSAImmutablize(
    Block* b,
    std::shared_ptr<BufferForest> buffer_forest) {
  auto nodes = b->nodes();
  for (auto node = nodes.front(); node != nodes.back();) {
    auto blocks = node->blocks();
    for (auto block : blocks) {
      TensorSSAImmutablize(block, buffer_forest);
    }

    if (immutable::immutableVersion.count(node->kind())) {
      if (buffer_forest->getBufferNodeOrNone(node->output())) {
        auto immutableNode = b->owningGraph()->create(
            c10::immutable::immutableVersion[node->kind()], node->inputs(), 1);
        immutableNode->output()->setType(node->output()->type());
        immutableNode->insertAfter(node);
        node->output()->replaceAllUsesWith(immutableNode->output());
        buffer_forest->replaceValue(node->output(), immutableNode->output());

        if (aten::copy_ == node->kind()) {
          buffer_forest->replaceMutation(node, immutableNode);
        }

        node->destroy();
        node = immutableNode->next();
      } else {
        node = node->next();
      }
    } else {
      node = node->next();
    }
  }
}

void DumbRemoveInterPrecedureMutation(std::shared_ptr<Graph> graph) {
  for (auto input : graph->inputs()) {
    WithInsertPoint p(graph->block()->param_node()->next());
    if (input->type()->cast<TensorType>()) {
      auto copy_node =
          graph
              ->create(aten::clone, {input, graph->insertConstant(IValue())}, 1)
              ->insertAfter(graph->block()->param_node()->next());
      copy_node->output()->copyMetadata(input);
      input->replaceAllUsesAfterNodeWith(copy_node, copy_node->output());
    }
  }
}

static void addMutatedValueToBlock(
    Value* mutated,
    Block* block,
    std::unordered_set<Block*>& visitedBlocks,
    std::shared_ptr<TensorSSAMutateInfo> mutateInfo,
    bool handleNode = true) {
  // Skip if this block if visited before
  if (visitedBlocks.count(block))
    return;
  visitedBlocks.insert(block);

  // Add to block and node returns
  block->insertOutput(block->outputs().size(), mutated);
  auto node = block->owningNode();
  if (handleNode) {
    auto nodeRet = node->addOutput();
    auto output_update =
        node->owningGraph()->create(tensorssa::Update, {nodeRet, mutated}, 0);
    output_update->insertAfter(node);
  }

  // Handle values that are specific to node kinds
  switch (node->kind()) {
    case prim::Loop: {
      // add to block parameter of loop body
      auto param = block->addInput();
      auto input_update =
          node->owningGraph()->create(tensorssa::Update, {param, mutated}, 0);
      input_update->insertBefore(block->nodes().front());
      // add to argument of loop node
      node->addInput(mutated);
      break;
    }

    case prim::If: {
      // add to the block of the other branch
      auto blockId = block == node->blocks()[1];
      addMutatedValueToBlock(
          mutated, node->blocks()[!blockId], visitedBlocks, mutateInfo, false);
      break;
    }
  }
}

void TensorSSAPropagation(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<TensorSSAMutateInfo> mutateInfo) {
  for (auto& mutated : mutateInfo->mutValues) {
    auto defBlock = mutated->node()->owningBlock();
    std::unordered_set<Block*> visitedBlocks;
    auto& nodes = mutateInfo->mutNodes[mutated];
    for (auto node : nodes) {
      for (auto block = node->owningBlock(); block != defBlock;
           block = block->owningNode()->owningBlock()) {
        addMutatedValueToBlock(mutated, block, visitedBlocks, mutateInfo);
      }
    }
  }
}

static void renameValues(Block* block) {
  for (auto node : block->nodes()) {
    if (tensorssa::Update == node->kind())
      node->input(1)->replaceAllUsesAfterNodeWith(node, node->input(0));
    for (auto& b : node->blocks())
      renameValues(b);
  }
}

void TensorSSARename(std::shared_ptr<Graph> graph) {
  // for (auto value : mutateInfo->mutValues)
  // mutateInfo->renameStacks.insert({value, {}});
  renameValues(graph->block());
}

void TensorSSARemoveUpdate(std::shared_ptr<Graph> graph) {
  std::function<void(Block*)> removeUpdateImpl;
  removeUpdateImpl = [&removeUpdateImpl](Block* b) -> void {
    auto nodes = b->nodes();
    for (auto node = nodes.front(); node != nodes.back();) {
      for (auto& block : node->blocks()) {
        removeUpdateImpl(block);
      }
      if (tensorssa::Update == node->kind()) {
        auto tmp = node->next();
        node->destroy();
        node = tmp;
      } else if (immutable::Assign == node->kind()) {
        node->output()->replaceAllUsesWith(node->input(1));
        auto tmp = node->next();
        node->destroy();
        node = tmp;
      } else {
        node = node->next();
      }
    }
  };
  removeUpdateImpl(graph->block());
}

void indexBoolFallback(std::shared_ptr<Graph> graph) {
  std::function<void(Block*)> indexBoolFallbackImpl;
  indexBoolFallbackImpl = [&indexBoolFallbackImpl](Block* b) -> void {
    auto nodes = b->nodes();
    for (auto node = nodes.front(); node != nodes.back();) {
      for (auto& block : node->blocks()) {
        indexBoolFallbackImpl(block);
      }
      if (immutable::Index == node->kind()) {
        auto indices = node->input(1)->node()->inputs();
        if (indices.front()->type()->cast<TensorType>()->scalarType() !=
            c10::kLong) {
          node->replaceWithNewSymbol(aten::index);
          node = node->next();
        }
      } else {
        node = node->next();
      }
    }
  };
  indexBoolFallbackImpl(graph->block());
}

void TensorSSARewriteMutation(
    std::shared_ptr<Graph> graph,
    std::shared_ptr<TensorSSAMutateInfo> mutateInfo) {
  // Preprocess: A dumb pass to eliminate interprecedure view
  // DumbRemoveInterPrecedureMutation(graph);

  // Step 0. convert inplace operator (add_, mul_, ...) to copy_
  RemoveInplace(graph);
  // ReplaceTensorToImmute(graph);

  // Step 1. Get Buffer Forest
  auto bufferForest = TensorSSAGetBufferForest(graph);

  // Step 2. Regularization `aten::view`, `aten::copy_` to
  // `immut::access`, `immut::assign`
  TensorSSAImmutablize(graph->block(), bufferForest);

  // Step 3. Convert to TensorSSA
  TensorSSAAliasRemoval(graph->block(), bufferForest, mutateInfo);
}

void ConvertToTensorSSA(std::shared_ptr<Graph> graph) {
  auto mutateInfo = std::make_shared<TensorSSAMutateInfo>();

  // Step 3. Convert to TensorSSA
  TensorSSARewriteMutation(graph, mutateInfo);

  // Step 4. block propogation
  // add alias block arguments
  TensorSSAPropagation(graph, mutateInfo);

  // Step 5. rename stack
  TensorSSARename(graph);

  // Step 6. fall back
  indexBoolFallback(graph);
}

} // namespace jit
} // namespace torch
