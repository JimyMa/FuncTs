#include <functs/csrc/jit/passes/convert_to_tensorssa.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void ConvertToTensorSSA(std::shared_ptr<Graph> graph) {
  std::cout << "Origin Graph: " << std::endl;
  graph->dump();

  // AliasDb
  auto aliasDb = AliasDb(graph);
  aliasDb.dump();
}

} // namespace jit
} // namespace torch
