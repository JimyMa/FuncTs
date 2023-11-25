#include <ATen/Context.h>
#include <ATen/ops/allclose.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/serialize.h>
// #include <torchvision/vision.h>

#include "passes/canonicalize.h"
#include "passes/common_passes.h"
#include "passes/freeze_module.h"
#include "passes/fuse_ops.h"
#include "passes/op_stat.h"
#include "passes/parallelize_loops.h"
#include "passes/refine_types.h"
#include "passes/te_op.h"
#include "passes/tensor_ssa.h"
#include "passes/type_utils.h"
#include "passes/unroll_loops.h"
#include "passes/validate_graph.h"
#include "run_utils.h"
#include "util/logging.h"
#include "util/rand.h"

using namespace torch::jit;

static void dumpGraphToFile(const std::shared_ptr<Graph> &graph,
                            const std::string &path) {
  if (!getenv("DUMP_GRAPH")) return;
  std::ofstream ofs(path);
  graph->print(ofs);
}

static IValue genValueOfType(const TypePtr &type) {
  switch (type->kind()) {
    case TypeKind::TensorType:
      return generateRandomTensor(type->cast<TensorType>());

    case TypeKind::TupleType: {
      auto elements = type->cast<TupleType>()->elements();
      c10::impl::GenericList list(getUnifiedElementType(type));
      for (auto &elem : elements) list.push_back(genValueOfType(elem));
      return list;
    }

    default:
      TORCH_CHECK(false, "Cannot generate input value for type ", *type);
  }
}

static Stack generateRandomInputs(const std::vector<TypePtr> &inputTypes) {
  Stack inputs;
  for (auto &type : inputTypes) inputs.push_back(genValueOfType(type));
  return inputs;
}

static std::string fmtIndices(const std::vector<size_t> &indices) {
  std::stringstream ss;
  print(ss, '[', indices.front(), ']');
  if (indices.size() == 1) return ss.str();
  ss << '[';
  for (auto i : c10::irange(1, indices.size())) {
    if (i > 1) ss << ", ";
    ss << indices[i];
  }
  ss << ']';
  return ss.str();
}

static void checkValue(const IValue &actual, const IValue &ref,
                       std::vector<size_t> &indices) {
  TORCH_CHECK(actual.tagKind() == ref.tagKind(), "Expect ", ref.tagKind(),
              ", got ", actual.tagKind(), " at ", fmtIndices(indices));
  if (actual.isTensor()) {
    auto actualTensor = actual.toTensor(), refTensor = ref.toTensor();
    TORCH_CHECK(actualTensor.sizes() == refTensor.sizes() &&
                    at::allclose(actualTensor, refTensor, 1e-3, 1e-5),
                "Inconsistent tensor at ", fmtIndices(indices),
                "\nReference: \n", refTensor, "\nActual: \n", actualTensor);
  } else if (actual.isList()) {
    auto realList = actual.toListRef(), refList = ref.toListRef();
    TORCH_CHECK(realList.size() == refList.size(), "Expect list of length ",
                refList.size(), ", got ", realList.size(), " at ",
                fmtIndices(indices));
    for (auto i : c10::irange(realList.size())) {
      indices.push_back(i);
      checkValue(realList[i], refList[i], indices);
      indices.pop_back();
    }
  } else if (actual.isTuple()) {
    auto &realTup = actual.toTupleRef().elements(),
         &refTup = ref.toTupleRef().elements();
    TORCH_CHECK(realTup.size() == refTup.size(), "Expect tuple of length ",
                refTup.size(), ", got ", realTup.size(), " at ",
                fmtIndices(indices));
    for (auto i : c10::irange(realTup.size())) {
      indices.push_back(i);
      checkValue(realTup[i], refTup[i], indices);
      indices.pop_back();
    }
  } else {
    TORCH_CHECK(actual == ref, "Unequal value at ", fmtIndices(indices));
  }
}

static void checkOutputs(const Stack &actualOutputs, const Stack &refOutputs) {
  TORCH_CHECK(actualOutputs.size() == refOutputs.size());
  std::vector<size_t> indices;
  for (auto i : c10::irange(refOutputs.size())) {
    auto &actualVal = actualOutputs[i], &refVal = refOutputs[i];
    indices.push_back(i);
    checkValue(actualVal, refVal, indices);
    indices.pop_back();
  }
}

static void dumpStruct(const IValue &val, size_t indent = 0) {
  for (auto _ : c10::irange(indent)) std::cout << "  ";
  for (auto &elem : val.toListRef()) dumpStruct(elem, indent + 1);
}

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: example <script-module> <input-types> <input-data>?\n";
    return 1;
  }
  // at::cuda::cudart
  // vision::cuda_version();
  at::globalContext().lazyInitCUDA();
  Module mod;
  try {
    mod = load(argv[1]);
  } catch (std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
  Freeze(&mod);
  auto graph = mod.get_method("forward").graph();
  ConvertProfilingInstrumentation(graph);
  dumpGraphToFile(graph, "after_freeze.rb");
  auto origin_graph = graph->copy();
  auto inputTypes = parseInputTypes(argv[2]);
  ValueTypeMap refinedTypes;
  try {
    RefineInputTypes(graph, inputTypes, refinedTypes);
    CanonicalizeOps(graph);
    if (getenv("PRINT_GRAPH_STAT")) CountMemoryIntensiveOps(graph);
    ToTensorSSA(graph);
    dumpGraphToFile(graph, "after_tssa.rb");
    ParallelizeLoops(graph);
    if (getenv("PRINT_GRAPH_STAT")) CountLoops(graph);
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
  } catch (std::exception &err) {
    std::cout << err.what();
    dumpGraphToFile(graph, "error.rb");
    return 1;
  }

  // Runtime
  c10::impl::GenericList dataset(AnyType::get());
  size_t numSamples = 0;
  if (argc > 3) {
    dataset = loadPickle<c10::impl::GenericList>(argv[3]);
    numSamples = dataset.size();
  } else {
    dataset.emplace_back(
        c10::ivalue::Tuple::create(generateRandomInputs(inputTypes)));
    numSamples = 1;
  }

  Code code(graph, "");
  GraphFunction origin_function("original", origin_graph, nullptr);

  Stack stack;
  for (auto i : c10::irange(numSamples)) {
    stack = getFeatureSample(dataset, i);
    torch::jit::InterpreterState(code).run(stack);
    auto output_tss_parallel = stack;
    stack = getFeatureSample(dataset, i);
    origin_function.run(stack);
    auto output_origin = stack;
    try {
      checkOutputs(output_tss_parallel, output_origin);
    } catch (std::exception &err) {
      // std::cout << "Inconsistency at sample " << i << '\n';
      // std::cerr << err.what();
    }
  }

  auto task = [&](size_t i) {
    auto stack = getFeatureSample(dataset, i % numSamples);
    torch::jit::InterpreterState(code).run(stack);
  };
  if (metricsEnabled()) {
    evalMetrics(task, numSamples);
  } else {
    auto result = evaluate(task);
    print(std::cout, "Latency: ", fmtDuration(result.mean()), '\n');
    printProfilingResults(result.count);
  }
}
