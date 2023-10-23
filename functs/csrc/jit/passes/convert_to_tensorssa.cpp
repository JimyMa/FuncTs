#include <c10/util/Exception.h>
#include <functs/csrc/jit/ir/alias_analysis.h>
#include <functs/csrc/jit/ir/symbol_ext.h>
#include <functs/csrc/jit/passes/convert_to_tensorssa.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/utils/memory_dag.h>

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
    const auto element = ptrPair.second;
    if (element->pointsTo.count() > 1) {
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

  aliasDb_buffer_tree.dump();
  aliasDb_buffer_tree.dumpToGraphvizFile("buffer_tree.dot");
}

void TensorSSAAliasRemoval(Block *b, AliasDbCopy *buffer_forest) {
  auto nodes = b->nodes();
  for (auto node : nodes) {
    auto blocks = node->blocks();
    for (auto block : blocks) {
      TensorSSAAliasRemoval(block, buffer_forest);
    }

    // Judge if a node is a write node, if true, convert to immutable equivalent
    // Step 1. Pass Up
    if (buffer_forest->isMutable(node)) {
      auto elementMap = buffer_forest->elementMapMutable();
      // TODO: only tensor values are considered...
      const Value *points_to;
      Value *leaf_value = node->output(0);
      Value *pass_up_value = node->input(0);
      b->owningGraph()->setInsertPoint(node->next());
      do {
        auto points_to_elem =
            buffer_forest->fromIndex(*elementMap[leaf_value]->pointsTo.begin());
        if (points_to_elem->values.size() > 0) {
          points_to = *(points_to_elem->values.begin());
          Node *pass_up_node;
          auto node = leaf_value->node();
          if (aten::select == node->kind() ||
              immutable::SelectImmutable == node->kind()) {
            pass_up_node = b->owningGraph()->create(
                immutable::SelectReverse, leaf_value->node()->inputs(), 1);
          } else if (aten::copy_ == node->kind() ||
                     immutable::Assign == node->kind()) {
            pass_up_node = b->owningGraph()->create(
                immutable::Assign, leaf_value->node()->inputs(), 1);
          } else {
            AT_ASSERT(false, "Unknown alias operator",
                      node->kind().toQualString());
          }
          pass_up_node->output(0)->copyMetadata(leaf_value);
          pass_up_node->replaceInput(0, pass_up_value);
          pass_up_value = pass_up_node->output(0);
          b->owningGraph()->insertNode(pass_up_node);
          b->owningGraph()->setInsertPoint(pass_up_node->next());
          leaf_value = const_cast<Value *>(points_to);
        } else {
          points_to = nullptr;
        }

      } while (points_to);

      std::cout << "leaf value: " << leaf_value->debugName() << std::endl;
      std::cout << "pass up value: " << pass_up_value->debugName() << std::endl;

      // Step 2. pass down
      // element = buffer_forest
    }
  }
}

void TensorSSAPropagation(std::shared_ptr<Graph> graph);
void TensorSSARename(std::shared_ptr<Graph> graph);

void ConvertToTensorSSA(std::shared_ptr<Graph> graph) {
  std::cout << "Origin Graph: " << std::endl;
  graph->dump();

  // AliasDb
  auto aliasDb_origin = AliasDbCopy(graph);
  auto aliasDb_buffer_tree = AliasDbCopy(graph);
  aliasDb_origin.dump();

  // Step 1. Get Buffer Tree
  GetBufferTree(graph, aliasDb_buffer_tree);

  // Step 2. Convert to TensorSSA
  TensorSSAAliasRemoval(graph->block(), &aliasDb_buffer_tree);
  graph->dump();
}

} // namespace jit
} // namespace torch
