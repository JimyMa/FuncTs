#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <memory>
#include <set>

#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <functs/csrc/jit/ir/alias_analysis.h>
#include <functs/csrc/jit/ir/buffer_forest.h>
#include <functs/csrc/jit/ir/symbol_ext.h>
#include <functs/csrc/jit/passes/convert_to_tensorssa.h>
#include <functs/csrc/jit/passes/remove_inplace.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/utils/memory_dag.h>
// #include <torch/csrc/jit/passes/utils/memory_dag.h>

using namespace c10;

namespace torch {
namespace jit {

void GetBufferTree(std::shared_ptr<Graph> g, AliasDbCopy &aliasDb_buffer_tree) {
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
        aliasDb_buffer_tree.mayAliasWildcard(value) /* wildcard node */) {
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

std::shared_ptr<BufferForest>
TensorSSAGetBufferForest(std::shared_ptr<Graph> graph) { // AliasDb
  auto aliasDb_buffer_tree = AliasDbCopy(graph);

  GetBufferTree(graph, aliasDb_buffer_tree);

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

void TensorSSAAliasRemoval(Block *b,
                           std::shared_ptr<BufferForest> functs_buffer_forest) {
  auto nodes = b->nodes();
  for (auto node = nodes.front(); node != nodes.back();) {
    auto blocks = node->blocks();
    for (auto block : blocks) {
      TensorSSAAliasRemoval(block, functs_buffer_forest);
    }

    // Judge if a node is a write node, if true, convert to immutable equivalent
    // Note: Unsupported alias has been eliminated from alias set
    if (functs_buffer_forest->isBufferMutation(node)) {
      // TODO: only tensor values are considered...
      // Step 1. Pass Up
      const Value *points_to;
      Value *leaf_value = node->output(0);
      Value *pass_up_value = node->input(0);
      WithInsertPoint insert_point(node->next());
      do {
        // substitute to buffer node
        auto leaf_buffer_node =
            functs_buffer_forest->getBufferTreeOrNone(leaf_value)
                ->getBufferNodeOrNone(leaf_value);
        auto points_to_node = leaf_buffer_node->pointsTo;

        // go up until meet the buffer root, the origin defined tensor who owns
        // the storage.
        if (points_to_node) {
          points_to = points_to_node->bufferNode_->var;
          Node *pass_up_node;
          auto leaf_node = leaf_value->node();
          if (aten::select == leaf_node->kind() ||
              immutable::Select == leaf_node->kind()) {
            pass_up_node = b->owningGraph()->create(
                immutable::SelectReverse, leaf_value->node()->inputs(), 1);
            pass_up_node->insertInput(1, pass_up_value);
          } else if (aten::copy_ == leaf_node->kind() ||
                     immutable::Assign == leaf_node->kind()) {
            pass_up_node = b->owningGraph()->create(
                immutable::Assign, leaf_value->node()->inputs(), 1);
          } else {
            AT_ASSERT(false, "Unknown alias operator when pass up",
                      leaf_node->kind().toQualString());
          }
          pass_up_node->output(0)->setType(leaf_value->type());
          pass_up_value = pass_up_node->output(0);
          b->owningGraph()->insertNode(pass_up_node);
          b->owningGraph()->setInsertPoint(pass_up_node->next());
          leaf_value = const_cast<Value *>(points_to);
        } else {
          points_to = nullptr;
        }
      } while (points_to);

      // Step 2. pass down
      // reconstruct elementMap_
      Value *pass_down_value = pass_up_value;
      Value *root_value = leaf_value;

      // functs_buffer_forest
      functs_buffer_forest->replaceValue(root_value, pass_down_value);

      root_value->replaceAllUsesAfterNodeWith(pass_up_value->node(),
                                              pass_down_value);

      WithInsertPoint pass_down_insert_point(pass_up_value->node()->next());

      std::function<void()> pass_down;

      pass_down = [&]() -> void {
        // auto elementMap = buffer_forest->elementMapMutable();
        auto pass_down_buffer_node =
            functs_buffer_forest->getBufferNodeOrNone(pass_down_value);
        for (auto point_from_buffer_node : pass_down_buffer_node->pointedFrom) {
          // auto from_elem = buffer_forest->fromIndex(pointedFromIndex);

          // auto from_value = const_cast<Value *>(*from_elem->values.begin());
          auto from_value = point_from_buffer_node->bufferNode_->var;
          auto from_node = from_value->node();

          if (!(from_node->isAfter(node))) {
            Node *pass_down_node;
            if (immutable::Select == from_node->kind()) {
              pass_down_node = b->owningGraph()->create(
                  immutable::Select, const_cast<Node *>(from_node)->inputs(),
                  1);
              pass_down_node->replaceInput(0, pass_down_value);
            } else if (aten::copy_ == node->kind() ||
                       immutable::Assign == node->kind()) {
              std::cout << "??SDFSDFSD" << std::endl;
              pass_down_node = b->owningGraph()->create(
                  immutable::Assign, const_cast<Node *>(from_node)->inputs(),
                  1);
              pass_down_node->replaceInput(0, pass_down_value);
              pass_down_node->replaceInput(1, pass_down_value);
            } else {
              AT_ASSERT(false, "Unknown alias operator when pass down",
                        from_node->kind().toQualString());
            }

            b->owningGraph()->insertNode(pass_down_node);
            b->owningGraph()->setInsertPoint(pass_down_node->next());
            pass_down_value = pass_down_node->output(0);
            pass_down_value->setType(
                const_cast<Node *>(from_node)->output(0)->type());

            functs_buffer_forest->replaceValue(from_value, pass_down_value);
            from_node->output(0)->replaceAllUsesAfterNodeWith(pass_down_node,
                                                              pass_down_value);

            root_value = from_node->output(0);
            pass_down();
          }
        }
      };
      pass_down();

      node->destroy();
      node = b->owningGraph()->insertPoint()->next();
    } else {
      node = node->next();
    }
  }
}

void TensorSSAPropagation(std::shared_ptr<Graph> graph);
void TensorSSARename(std::shared_ptr<Graph> graph);

void TensorSSAImmutablize(Block *b,
                          std::shared_ptr<BufferForest> buffer_forest) {
  auto nodes = b->nodes();
  for (auto node = nodes.front(); node != nodes.back();) {
    auto blocks = node->blocks();
    for (auto block : blocks) {
      TensorSSAImmutablize(block, buffer_forest);
    }
    if (buffer_forest->getBufferNodeOrNone(node->output())) {
      if (aten::select == node->kind()) {
        auto immutableNode =
            b->owningGraph()->create(immutable::Select, node->inputs(), 1);
        immutableNode->output()->setType(node->output()->type());
        immutableNode->insertAfter(node);
        node->output()->replaceAllUsesWith(immutableNode->output());
        buffer_forest->replaceValue(node->output(), immutableNode->output());
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

void TensorSSABufferTreeViewImmutablize(
    Block *b, std::shared_ptr<BufferForest> functs_buffer_forest) {

  auto nodes = b->nodes();

  for (auto node = nodes.front(); node != nodes.back();) {
    auto blocks = node->blocks();
    for (auto &block : blocks) {
      TensorSSABufferTreeViewImmutablize(block, functs_buffer_forest);
    }

    auto output_value = node->output();
    auto buffer_node = functs_buffer_forest->getBufferNodeOrNone(output_value);

    if (buffer_node) {
      if (aten::select == node->kind()) {
        // add a new immutable operator
        auto new_node =
            b->owningGraph()->create(immutable::Select, node->inputs(), 1);

        new_node->output()->setType(node->output()->type());
        new_node->insertBefore(node);
        node->output()->replaceAllUsesWith(new_node->output());

        functs_buffer_forest->replaceValue(output_value, new_node->output());
        node->dump();
        new_node->dump();
        node->destroy();
        node = new_node->next();
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
      input->replaceAllUsesWith(copy_node->output());
    }
  }
}

void ConvertToTensorSSA(std::shared_ptr<Graph> graph) {
  std::cout << "Origin Graph: " << std::endl;
  graph->dump();

  // Preprocess: A dumb pass to eliminate interprecedure view
  std::cout << "dumb remove inter precedure mutation begin..." << std::endl;
  DumbRemoveInterPrecedureMutation(graph);
  std::cout << "dumb remove inter precedure mutation end..." << std::endl;

  // Step 0. convert inplace operator (add_, mul_, ...) to copy
  std::cout << "remove inplace begin..." << std::endl;
  RemoveInplace(graph);
  std::cout << "remove inplace end..." << std::endl;

  // Step 1. Get Buffer Forest
  std::cout << "get Buffer Tree begin..." << std::endl;
  auto bufferForest = TensorSSAGetBufferForest(graph);
  std::cout << "get Buffer Tree end..." << std::endl;

  bufferForest->dump();

  // Step 2. Regularization `aten::view`, `aten::copy_` to
  // `immut::access`, `immut::assign`
  TensorSSAImmutablize(graph->block(), bufferForest);
  // graph->dump();
  bufferForest->dump();

  // Step 2. Convert to TensorSSA
  // LOG(INFO) << "Step 2. Functionaliazation" << std::endl;
  TensorSSAAliasRemoval(graph->block(), bufferForest);

  graph->dump();
}

} // namespace jit
} // namespace torch
