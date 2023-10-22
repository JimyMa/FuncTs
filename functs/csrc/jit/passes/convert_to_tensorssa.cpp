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
  auto aliasDb = AliasDbCopy(graph);
  aliasDb.dump();

  // ConvertToTensorSSAImpl(graph->block(), &aliasDb);
  // auto elementMap = aliasDb.elementMap();
  // auto memoryDAG = aliasDb.memoryDAG();
}

} // namespace jit
} // namespace torch
