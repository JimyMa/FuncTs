#include <torch/csrc/jit/serialization/import.h>
#include <torch/serialize.h>
#include <torch_tensorrt/torch_tensorrt.h>
// #include <torchvision/vision.h>

#include "run_utils.h"

using namespace torch::jit;
using namespace torch_tensorrt;
using namespace torch_tensorrt::torchscript;
using namespace std::chrono;
using namespace std::chrono_literals;

int main(int argc, char const *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: run_net_trt <script-module> <input-data>\n";
    return 1;
  }
  Module mod;
  try {
    mod = load(argv[1]);
    mod.to(c10::kCUDA);
  } catch (std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
  auto dataset = loadPickle<at::Tensor>(argv[2]).cuda();
  auto numSamples = size_t(dataset.size(0));
  CompileSpec spec({Input(dataset.slice(0, 0, 1))});
  spec.enabled_precisions = {c10::kFloat, c10::kHalf};
  mod = compile(mod, std::move(spec));
  {
    auto result = evaluate([&](size_t i) {
      i %= numSamples;
      mod.forward({dataset.slice(0, i, i + 1)});
    });
    print(std::cout, "Latency: ", fmtDuration(result.mean()), '\n');
  }
}
