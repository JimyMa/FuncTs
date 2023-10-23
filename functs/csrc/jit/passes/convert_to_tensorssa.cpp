#include <functs/csrc/jit/ir/alias_analysis.h>
#include <functs/csrc/jit/passes/convert_to_tensorssa.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// void ConvertToTensorSSAImpl(Block *block, AliasDbCopy *db) {
//   auto nodes = block->nodes();
//   for (auto it = nodes.begin(); it != nodes.end(); ++it) {
//     Node *node = *it;
//     for (Block *subblock : node->blocks()) {
//       ConvertToTensorSSAImpl(subblock, db);
//     }
//     if (db->isMutable(node)) {
//       std::cout << "mutable node: " << std::endl;
//     }
//     node->dump();
//   }
// }

void ConvertToTensorSSA(std::shared_ptr<Graph> graph) {
  std::cout << "Origin Graph: " << std::endl;
  graph->dump();

  // AliasDb
  auto aliasDb_origin = AliasDbCopy(graph);
  auto aliasDb_intra_procedure = AliasDbCopy(graph);
  aliasDb_origin.dump();

  // ConvertToTensorSSAImpl(graph->block(), &aliasDb);
  auto elementMap = aliasDb_origin.elementMap();

  for (const auto &ptrPair : elementMap) {
    const auto element = ptrPair.second;
    int ct = 0;
    if (!element->pointsTo.empty()) {
      auto begin_name = aliasDb_origin.getElementName(element);
      std::cout << "begin name: " << begin_name << std::endl;

      for (const auto pointedTo : element->pointsTo) {
        auto end_name =
            aliasDb_origin.getElementName(aliasDb_origin.fromIndex(pointedTo));
        std::cout << "end name: " << end_name << std::endl;
      }
    }
    ct = 0;
  }
}

} // namespace jit
} // namespace torch
