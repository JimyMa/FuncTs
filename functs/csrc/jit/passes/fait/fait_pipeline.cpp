#include <ATen/core/jit_type_base.h>
#include <functs/csrc/jit/ir/symbol_ext.h>
#include <functs/csrc/jit/passes/fait/fait_pipeline.h>
#include <passes/type_utils.h>
#include <unordered_map>

namespace torch {
namespace jit {

static void dumpGraphToFile(
    const std::shared_ptr<Graph>& graph,
    const std::string& path) {
  if (!getenv("DUMP_GRAPH"))
    return;
  std::ofstream ofs(path);
  graph->print(ofs, false);
}

void FaitGetRefineType(
    const std::shared_ptr<Graph> graph,
    std::vector<c10::TypePtr> type_hint,
    std::unordered_map<Value*, TypePtr>& refinedTypes) {
  // vision::cuda_version();
  // ConvertProfilingInstrumentation(graph);
  RefineInputTypes(graph, type_hint, refinedTypes);
  CanonicalizeOps(graph);
  InferDtypeAndDevice(graph, refinedTypes);
  InferShape(graph, refinedTypes);
}

void FaitPipeline(
    const std::shared_ptr<Graph> graph,
    std::vector<c10::TypePtr> type_hint) {
  // auto graph = module.get_method("forward").graph()->copy();
  // vision::cuda_version();
  std::unordered_map<Value*, TypePtr> refinedTypes;
  // ConvertProfilingInstrumentation(graph);
  RefineInputTypes(graph, type_hint, refinedTypes);
  CanonicalizeOps(graph);
  if (getenv("PRINT_GRAPH_STAT"))
    CountMemoryIntensiveOps(graph);
  // ToTensorSSA(graph);
  dumpGraphToFile(graph, "after_tssa.rb");
  ParallelizeLoops(graph);
  if (getenv("PRINT_GRAPH_STAT"))
    CountLoops(graph);
  InferDtypeAndDevice(graph, refinedTypes);
  InferShape(graph, refinedTypes);
  dumpGraphToFile(graph, "after_par.rb");
  FuseOps(graph, refinedTypes);
  dumpGraphToFile(graph, "after_fuse.rb");
  UnrollLoopsWithDeps(graph, refinedTypes);
  UnrollSimpleMaps(graph, refinedTypes);
  InferShape(graph, refinedTypes);
  FuseOps(graph, refinedTypes);
  dumpGraphToFile(graph, "after_unroll.rb");

  SplitParallelMaps(graph, refinedTypes);
  dumpGraphToFile(graph, "after_split.rb");
  ToMutableTensors(graph);
  ConvertInfusibleMapsToLoops(graph, refinedTypes);
  CanonicalizeFusableMaps(graph);
  dumpGraphToFile(graph, "after_back.rb");
  MapFunctorToParallelization(graph, refinedTypes);
  FusedOpToParallelization(graph, refinedTypes);
  dumpGraphToFile(graph, "after_codegen.rb");
  Validate(graph);
}
} // namespace jit
} // namespace torch
