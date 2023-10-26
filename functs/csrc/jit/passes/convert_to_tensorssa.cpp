#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
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

struct TensorSSAMutateInfo {
  std::vector<Value *> mutValues{};
  std::unordered_map<Value *, std::vector<Node *>> mutNodes{};
  std::unordered_map<Value *, Value *> valueToMut{};
  std::unordered_map<Value *, std::vector<Value *>> renameStacks{};

  void addMutNodes(Node *updateNode) {
    AT_ASSERT(tensorssa::Update == updateNode->kind(),
              "don't forget we use update node to annotate mutation");

    auto mutated = updateNode->input(1);
    auto mutatee = updateNode->input(0);

    if (!mutNodes.count(mutated)) {
      mutValues.push_back(mutated);
      mutNodes.insert({mutated, std::vector<Node *>()});
    }
    valueToMut.insert({mutatee, mutated});
    mutNodes[mutated].push_back(updateNode);
  }
};

static void GetBufferTreeAliasDb(std::shared_ptr<Graph> g,
                                 AliasDbCopy &aliasDb_buffer_tree) {
  // Note: Unsupported alias relationship by now:
  // // - An element points to more than one elements (i.e. a <- c, b <- c)
  // TODO: Cannot support add Container alias analysis by now
  // Note: Only intra-procedure are supported by now

  // g: Graph-level IR based tensor program
  // alias_db_origin: origin alias relationship
  // alias_db_buffer_tree: modified alias relationship

  // Step 1. visit all elements in aliasdb, delete all unsupported alias
  // relationship.
  // // 1. find all ambigious alias relationship
  // // 2. delete them from aliasDb_buffer_tree

  bool any_changed = true;
  auto elementMap = aliasDb_buffer_tree.elementMapMutable();

  ska::flat_hash_map<const Value *, Element *> ambigious_alias;

  // Step 1
  for (const auto &ptrPair : elementMap) {
    auto value = ptrPair.first;
    auto element = ptrPair.second;

    if (element->pointsTo.count() > 1 /* Step 1.1: count > 1 */ ||
        aliasDb_buffer_tree.mayAliasWildcard(value) /* wildcard node */ ||
        prim::Loop == value->node()->kind()) {
      ambigious_alias[ptrPair.first] = ptrPair.second;
    }
  }

  ska::flat_hash_map<const Value *, Element *> may_alias;

  // Step 2
  // get may_alias
  for (const auto &ptrPair : elementMap) {
    for (const auto &ptrPairAmbigious : ambigious_alias) {
      if (aliasDb_buffer_tree.mayAlias(ptrPair.first, ptrPairAmbigious.first) &&
          !may_alias.count(ptrPair.first)) {
        may_alias.insert(ptrPair);
      }
    }
  }

  // Delete from elementMap
  for (const auto &ptrPair : may_alias) {
    aliasDb_buffer_tree.elementMapErase(ptrPair.first);
  }
}

static std::shared_ptr<BufferForest>
TensorSSAGetBufferForest(std::shared_ptr<Graph> graph) { // AliasDb
  auto aliasDb_buffer_tree = AliasDbCopy(graph);

  GetBufferTreeAliasDb(graph, aliasDb_buffer_tree);

  auto bufferForest = std::make_shared<BufferForest>();
  auto elementMap = aliasDb_buffer_tree.elementMap();
  for (auto &elemPtr : elementMap) {
    auto value = elemPtr.first;
    auto elem = elemPtr.second;
    for (auto pointToIndex : elem->pointsTo) {
      bufferForest->addEdgeToBufferForest(
          const_cast<Value *>(value),
          const_cast<Value *>(
              *aliasDb_buffer_tree.fromIndex(pointToIndex)->values.begin()));
    }
  }
  auto writeIndex = aliasDb_buffer_tree.writeIndexMutable();
  for (auto node_vs_idx : *writeIndex) {
    bufferForest->addMutationToBufferForest(node_vs_idx.first);
  }
  return bufferForest;
}

static void
TensorSSAAliasRemoval(Block *b, std::shared_ptr<BufferForest> bufferForest,
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
      b->owningGraph()->dump();
      bufferForest->dump();
      // TODO: only tensor values are considered...
      // Step 1. Pass Up
      const Value *points_to;
      Value *leaf_value = node->output(0);
      Value *pass_up_value = node->input(0);
      Node *node_insert = node;
      do {
        // substitute to buffer node
        auto leaf_buffer_node = bufferForest->getBufferTreeOrNone(leaf_value)
                                    ->getBufferNodeOrNone(leaf_value);
        auto points_to_node = leaf_buffer_node->pointsTo;

        // go up until meet the buffer root, the origin defined tensor who owns
        // the storage.
        if (points_to_node) {
          points_to = points_to_node->bufferNode_->var;
          Node *pass_up_node;
          auto leaf_node = leaf_value->node();
          if (immutable::Select == leaf_node->kind()) {
            pass_up_node = b->owningGraph()->create(
                immutable::SelectReverse, leaf_value->node()->inputs(), 1);
            pass_up_node->insertInput(1, pass_up_value);
          } else if (immutable::Assign == leaf_node->kind()) {
            pass_up_node = b->owningGraph()->create(
                immutable::Assign, leaf_value->node()->inputs(), 1);
          } else {
            AT_ASSERT(false, "Unknown alias operator when pass up",
                      leaf_node->kind().toQualString());
          }
          pass_up_node->output(0)->setType(leaf_value->type());
          pass_up_value = pass_up_node->output(0);

          pass_up_node->insertAfter(node_insert);
          node_insert = pass_up_node;
          leaf_value = const_cast<Value *>(points_to);
        } else {
          points_to = nullptr;
        }
      } while (points_to);

      // Step 2. pass down
      // reconstruct elementMap_
      Value *pass_down_value = pass_up_value;
      Node *pass_down_node = pass_down_value->node();
      Value *root_value = leaf_value;
      node_insert = pass_down_value->node();

      // generate a strong update to beacon mutation
      auto update_node = b->owningGraph()->create(
          tensorssa::Update, {pass_down_node->output(), root_value}, 0);
      update_node->insertAfter(node_insert);
      node_insert = update_node;

      mutateInfo->addMutNodes(update_node);

      std::function<void()> pass_down;

      pass_down = [&]() -> void {
        // auto elementMap = buffer_forest->elementMapMutable();
        auto pass_down_buffer_node =
            bufferForest->getBufferNodeOrNone(root_value);
        for (auto point_from_buffer_node : pass_down_buffer_node->pointedFrom) {
          // auto from_elem = buffer_forest->fromIndex(pointedFromIndex);

          // auto from_value = const_cast<Value *>(*from_elem->values.begin());
          auto from_value = point_from_buffer_node->bufferNode_->var;
          auto from_node = from_value->node();

          if (from_node->isBefore(node) && node->isDominatedBy(from_node)) {
            if (immutable::Select == from_node->kind()) {
              pass_down_node = b->owningGraph()->create(
                  immutable::Select, const_cast<Node *>(from_node)->inputs(),
                  1);
              pass_down_node->replaceInput(0, pass_down_value);
            } else if (aten::copy_ == node->kind() ||
                       immutable::Assign == node->kind()) {
              pass_down_node = b->owningGraph()->create(
                  immutable::Assign, const_cast<Node *>(from_node)->inputs(),
                  1);
              pass_down_node->replaceInput(0, pass_down_value);
              pass_down_node->replaceInput(1, pass_down_value);
            } else {
              AT_ASSERT(false, "Unknown alias operator when pass down",
                        from_node->kind().toQualString());
            }

            // b->owningGraph()->insertNode(pass_down_node);
            pass_down_node->insertAfter(node_insert);
            pass_down_value = pass_down_node->output(0);
            pass_down_value->copyMetadata(from_node->output(0));

            // generate a strong update to beacon mutation
            auto update_node = b->owningGraph()->create(
                tensorssa::Update, {pass_down_node->output(), from_value}, 0);
            update_node->insertAfter(pass_down_node);
            node_insert = update_node;

            mutateInfo->addMutNodes(update_node);

            root_value = from_node->output(0);
            pass_down();
          }
        }
      };
      pass_down();
      node = node_insert->next();
    } else {
      node = node->next();
    }
  }
}

void TensorSSAImmutablize(Block *b,
                          std::shared_ptr<BufferForest> buffer_forest) {
  auto nodes = b->nodes();
  for (auto node = nodes.front(); node != nodes.back();) {
    auto blocks = node->blocks();
    for (auto block : blocks) {
      TensorSSAImmutablize(block, buffer_forest);
    }

    switch (node->kind()) {
    case aten::select:
    case aten::copy_: {
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
      break;
    }
    default:
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
    Value *mutated, Block *block, std::unordered_set<Block *> &visitedBlocks,
    std::shared_ptr<TensorSSAMutateInfo> mutateInfo, bool handleNode = true) {
  // Skip if this block if visited before
  if (visitedBlocks.count(block))
    return;
  visitedBlocks.insert(block);

  // Add to block and node returns
  block->insertOutput(block->outputs().size(), mutated);
  auto node = block->owningNode();
  if (handleNode) {
    auto nodeRet = node->addOutput();
    mutateInfo->valueToMut.insert({nodeRet, mutated});
  }

  // Handle values that are specific to node kinds
  switch (node->kind()) {
  case prim::Loop: {
    // add to block parameter of loop body
    auto param = block->addInput();
    mutateInfo->valueToMut.insert({param, mutated});
    // add to argument of loop node
    node->addInput(mutated);
    break;
  }

  case prim::If: {
    // add to the block of the other branch
    auto blockId = block == node->blocks()[1];
    addMutatedValueToBlock(mutated, node->blocks()[!blockId], visitedBlocks,
                           mutateInfo, false);
    break;
  }
  }
}

static void
TensorSSAPropagation(std::shared_ptr<Graph> graph,
                     std::shared_ptr<TensorSSAMutateInfo> mutateInfo) {

  for (auto &mutated : mutateInfo->mutValues) {
    mutateInfo->valueToMut.insert({mutated, mutated});
    auto defBlock = mutated->node()->owningBlock();
    std::unordered_set<Block *> visitedBlocks;
    auto &nodes = mutateInfo->mutNodes[mutated];
    for (auto node : nodes) {
      for (auto block = node->owningBlock(); block != defBlock;
           block = block->owningNode()->owningBlock()) {
        addMutatedValueToBlock(mutated, block, visitedBlocks, mutateInfo);
      }
    }
  }
}

static void renameValues(Block *block,
                         std::shared_ptr<TensorSSAMutateInfo> mutateInfo) {
  // Initialize rename counts in current scope
  std::unordered_map<Value *, size_t> renameCounts;
  auto updateValue = [&](Value *value) {
    // find mutated version of this value
    Value *mutated = nullptr;
    if (mutateInfo->valueToMut.count(value)) {
      mutated = mutateInfo->valueToMut[value];
    } else {
      auto defNode = value->node();
      auto kind = defNode->kind();
      if (kind == tensorssa::Update) {
        mutated = mutateInfo->valueToMut[defNode->input(0)];
        mutateInfo->valueToMut.insert({value, mutated});
      }
    }
    if (!mutated)
      return;

    // add to rename stack
    mutateInfo->renameStacks[mutated].push_back(value);
    // add to rename counts
    if (renameCounts.count(mutated))
      renameCounts[mutated]++;
    else
      renameCounts.insert({mutated, 1});
  };
  auto replaceInputsOf = [&](Node *node) -> void {
    if (tensorssa::Update == node->kind())
      return;
    for (auto i = 0u; i < node->inputs().size(); i++) {
      auto input = node->input(i);
      if (!mutateInfo->valueToMut.count(input))
        continue;
      auto mutated = mutateInfo->valueToMut[input];
      auto latest = mutateInfo->renameStacks[mutated].back();
      node->replaceInput(i, latest);
    }
  };

  // Add parameters to rename stack
  for (auto param : block->inputs())
    updateValue(param);

  // Process each node
  for (auto node : block->nodes()) {
    // replace inputs
    replaceInputsOf(node);
    // visit owned blocks
    for (auto nested : node->blocks())
      renameValues(nested, mutateInfo);
    // update outputs
    for (auto output : node->outputs())
      updateValue(output);
  }

  // Process return node
  replaceInputsOf(block->return_node());

  // Restore rename stack
  for (auto &pair : renameCounts) {
    for (auto i = 0u; i < pair.second; i++)
      mutateInfo->renameStacks[pair.first].pop_back();
  }
}

static void TensorSSARename(std::shared_ptr<Graph> graph,
                            std::shared_ptr<TensorSSAMutateInfo> mutateInfo) {
  for (auto value : mutateInfo->mutValues)
    mutateInfo->renameStacks.insert({value, {}});
  renameValues(graph->block(), mutateInfo);
}

static void TensorSSARemoveUpdate(std::shared_ptr<Graph> graph) {}

void ConvertToTensorSSA(std::shared_ptr<Graph> graph) {
  std::cout << "Origin Graph: " << std::endl;
  graph->dump();

  // Preprocess: A dumb pass to eliminate interprecedure view
  std::cout << "dumb remove inter precedure mutation begin..." << std::endl;
  DumbRemoveInterPrecedureMutation(graph);
  std::cout << "dumb remove inter precedure mutation end..." << std::endl;
  graph->dump();

  // Step 0. convert inplace operator (add_, mul_, ...) to copy
  std::cout << "remove inplace begin..." << std::endl;
  RemoveInplace(graph);
  std::cout << "remove inplace end..." << std::endl;

  // Step 1. Get Buffer Forest
  std::cout << "get Buffer Tree begin..." << std::endl;
  auto bufferForest = TensorSSAGetBufferForest(graph);
  std::cout << "get Buffer Tree end..." << std::endl;

  // Step 2. Regularization `aten::view`, `aten::copy_` to
  // `immut::access`, `immut::assign`
  TensorSSAImmutablize(graph->block(), bufferForest);
  graph->dump();
  bufferForest->dump();

  auto mutateInfo = std::make_shared<TensorSSAMutateInfo>();
  // Step 3. Convert to TensorSSA
  // LOG(INFO) << "Step 2. Functionaliazation" << std::endl;
  std::cout << "Tensor Alias Removal begin..." << std::endl;
  TensorSSAAliasRemoval(graph->block(), bufferForest, mutateInfo);
  std::cout << "Tensor Alias Removal end..." << std::endl;
  graph->dump();

  // Step 4. block propogation
  // add alias block arguments
  /////////////////////////////////
  //  IMPLEMENTED BY ZIHAN WANG  //
  //            YYDS             //
  /////////////////////////////////
  std::cout << "Tensor Propagation begin..." << std::endl;
  TensorSSAPropagation(graph, mutateInfo);
  std::cout << "Tensor Propagation end..." << std::endl;
  graph->dump();

  // Step 5. rename stack
  // rename by stack
  std::cout << "Tensor rename begin..." << std::endl;
  TensorSSARename(graph, mutateInfo);
  std::cout << "Tensor rename end..." << std::endl;
  graph->dump();
}

} // namespace jit
} // namespace torch
