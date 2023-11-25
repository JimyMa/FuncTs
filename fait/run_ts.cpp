#include <ATen/Context.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/serialize.h>
// #include <torchvision/vision.h>

#include "run_utils.h"

using namespace torch::jit;

int main(int argc, const char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage: example <script-module> <input-types> <input-data>\n";
    return 1;
  }
  // vision::cuda_version();
  at::globalContext().lazyInitCUDA();
  Module mod;
  try {
    mod = load(argv[1]);
  } catch (std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
  freeze_module_inplace(&mod);
  auto graph = mod.get_method("forward").graph();
  ConvertProfilingInstrumentation(graph);

  // Runtime
  c10::impl::GenericList dataset(AnyType::get());
  dataset = loadPickle<c10::impl::GenericList>(argv[3]);
  size_t numSamples = dataset.size();

  GraphFunction function("original", graph, nullptr);

  Stack stack;
  for (auto i : c10::irange(numSamples)) {
    stack = getFeatureSample(dataset, i);
    function.run(stack);
  }

  auto task = [&](size_t i) {
    auto stack = getFeatureSample(dataset, i % numSamples);
    function.run(stack);
  };
  if (metricsEnabled()) {
    evalMetrics(task, numSamples);
  } else {
    auto result = evaluate(task);
    print(std::cout, "Latency: ", fmtDuration(result.mean()), '\n');
    printProfilingResults(result.count);
  }
}

