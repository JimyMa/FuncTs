#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/serialize.h>
#include <torch_tensorrt/torch_tensorrt.h>
// #include <torchvision/vision.h>

#include "passes/refine_types.h"
#include "run_utils.h"

using namespace torch::jit;
using namespace torch_tensorrt;
using namespace torch_tensorrt::torchscript;
using namespace std::chrono;
using namespace std::chrono_literals;

static void flattenForwardInputs(Module mod,
                                 const std::vector<TypePtr> &inputTypes) {
  // Rewrite graph inputs
  auto method = mod.get_method("forward");
  auto graph = method.graph();
  auto graphInputs = graph->inputs().vec();
  WithInsertPoint insertPt(graph->block()->nodes().front());
  size_t realInputIdx = 1;
  for (auto inputIdx : c10::irange(inputTypes.size())) {
    auto input = graphInputs[inputIdx + 1];
    auto type = inputTypes[inputIdx];
    if (type->kind() == TypeKind::TupleType) {
      auto elemTypes = type->cast<TupleType>()->elements();
      std::vector<Value *> params;
      for (auto elemIdx : c10::irange(elemTypes.size())) {
        auto elemTy = elemTypes[elemIdx];
        auto inputName = input->debugName() + "_" + std::to_string(elemIdx);
        auto param = graph->insertInput(realInputIdx++, inputName);
        params.push_back(param);
      }
      auto listNode = graph->insertNode(graph->createList(
          input->type()->cast<ListType>()->getElementType(), params));
      input->replaceAllUsesWith(listNode->output(0));
      graph->eraseInput(realInputIdx);
    } else {
      realInputIdx++;
    }
  }

  // Rewrite function schema
  auto &func = method.function();
  std::vector<Argument> schemaArgs;
  for (auto input : graph->inputs())
    schemaArgs.emplace_back(input->debugName(), input->type());
  func.setSchema(func.getSchema().cloneWithArguments(std::move(schemaArgs)));
}

static void collectSpec(TypePtr type, std::vector<Input> &specs) {
  switch (type->kind()) {
    case TypeKind::TensorType: {
      auto tensorTy = type->cast<TensorType>();
      specs.emplace_back(*tensorTy->sizes().concrete_sizes(),
                         *tensorTy->scalarType());
      return;
    } break;

    case TypeKind::TupleType: {
      auto elems = type->cast<TupleType>()->elements();
      for (auto &elem : elems) collectSpec(elem, specs);
    } break;

    default:
      TORCH_CHECK(false, "Cannot collect specification for type ", *type);
  }
}

static CompileSpec getFlattenedSpec(const std::vector<TypePtr> &inputTypes) {
  std::vector<Input> specs;
  for (auto &type : inputTypes) collectSpec(type, specs);
  return std::move(specs);
}

static void flattenIValue(const IValue &ival, Stack &inputs) {
  if (ival.isTensor()) {
    inputs.push_back(ival.toTensor().cuda());
  } else if (ival.isTuple()) {
    auto &tup = ival.toTupleRef().elements();
    for (auto &elem : tup) flattenIValue(elem, inputs);
  } else if (ival.isList()) {
    auto list = ival.toListRef();
    for (auto &elem : list) flattenIValue(elem, inputs);
  } else {
    TORCH_CHECK(false, "Cannot handle input value of tag ", ival.tagKind());
  }
}

static Stack getFlattenedSample(const c10::List<IValue> &dataset,
                                size_t index) {
  auto &tup = dataset.get(index).toTupleRef().elements();
  Stack inputs;
  for (auto &elem : tup) flattenIValue(elem, inputs);
  return std::move(inputs);
}

int main(int argc, char const *argv[]) {
  if (argc < 4) {
    std::cerr << "usage: run_trt <script-module> <input-types> <input-data>\n";
    return 1;
  }
  Module mod;
  try {
    mod = load(argv[1]);
  } catch (std::exception &e) {
    std::cerr << e.what();
    return 1;
  }

  auto inputTypes = parseInputTypes(argv[2]);
  flattenForwardInputs(mod, inputTypes);
  freeze_module_inplace(&mod);
  ConvertProfilingInstrumentation(mod.get_method("forward").graph());

  auto spec = getFlattenedSpec(inputTypes);
  auto dataset = loadPickle<c10::impl::GenericList>(argv[3]);
  auto numSamples = dataset.size();
  auto stack = getFlattenedSample(dataset, 0);
  spec.torch_executed_ops = {"prim::ListConstruct", "prof::Begin", "prof::End"};
  spec.truncate_long_and_double = true;
  mod = compile(mod, std::move(spec));

  for (auto i : c10::irange(numSamples)) {
    auto stack = getFlattenedSample(dataset, i % numSamples);
    mod.forward(stack);
  }

  auto task = [&](size_t i) {
    auto stack = getFlattenedSample(dataset, i % numSamples);
    mod.forward(stack);
  };
  if (metricsEnabled()) {
    evalMetrics(task, numSamples);
  } else {
    auto result = evaluate(task);
    print(std::cout, "Latency: ", fmtDuration(result.mean()), '\n');
    printProfilingResults(result.count);
  }
}
