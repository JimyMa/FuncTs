#include <ATen/core/jit_type_base.h>
#include <functs/csrc/jit/ir/symbol_ext.h>
#include <functs/csrc/jit/passes/fait/fait_pipeline.h>

namespace torch {
namespace jit {

void FaitPipeline(std::shared_ptr<Graph> graph,
                  std::vector<c10::TypePtr> type_hint) {
  vision::cuda_version();
  std::unordered_map<Value *, TypePtr> refinedTypes;
  // ConvertProfilingInstrumentation(graph);
  RefineInputTypes(graph, type_hint, refinedTypes);
  CanonicalizeOps(graph);
  ParallelizeLoops(graph);
  InferDtypeAndDevice(graph, refinedTypes);
  InferShape(graph, refinedTypes);

  FuseOps(graph, refinedTypes);

  UnrollLoopsWithDeps(graph, refinedTypes);
  UnrollSimpleMaps(graph, refinedTypes);
  InferShape(graph, refinedTypes);
  FuseOps(graph, refinedTypes);

  SplitParallelMaps(graph, refinedTypes);

  ConvertInfusibleMapsToLoops(graph, refinedTypes);
  CanonicalizeFusableMaps(graph);

  MapFunctorToParallelization(graph, refinedTypes);
  FusedOpToParallelization(graph, refinedTypes);

  Validate(graph);
}
} // namespace jit
} // namespace torch
