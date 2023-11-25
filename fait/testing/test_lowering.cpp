#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/randint.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
// #include <torchvision/vision.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "passes/fuse_ops.h"
#include "passes/te_op.h"
#include "util/common.h"
#include "util/logging.h"
#include "util/types.h"

using json = nlohmann::json;
using namespace torch::jit;

static Value *createValue(TypePtr type, const json &input, Graph *graph) {
  switch (type->kind()) {
    case TypeKind::IntType: {
      return graph->insertConstant(input.get<int64_t>());
    } break;

    case TypeKind::BoolType: {
      return graph->insertConstant(input.get<bool>());
    } break;

    case TypeKind::FloatType: {
      return graph->insertConstant(input.get<float>());
    } break;

    case TypeKind::NumberType: {
      IValue value;
      if (input.is_number_float())
        value = input.get<float>();
      else if (input.is_number_integer())
        value = input.get<int64_t>();
      else
        TORCH_CHECK(false, "Cannot convert ", input, " to number");
      return graph->insertConstant(value);
    } break;

    case TypeKind::TensorType: {
      auto shape = input.at("shape").get<std::vector<int64_t>>();
      auto dtype = strToDtype.at(input.at("dtype").get<std::string>());
      return graph->addInput()->setType(
          TensorType::createContiguous(dtype, c10::kCUDA, shape));
    } break;

    case TypeKind::ListType: {
      auto elemType = type->cast<ListType>()->getElementType();
      auto elemJsons = input.get<std::vector<json>>();
      std::vector<Value *> elemValues;
      for (auto &elem : elemJsons)
        elemValues.push_back(createValue(elemType, elem, graph));
      auto list = graph->appendNode(graph->createList(elemType, elemValues));
      return list->output(0);
    } break;

    case TypeKind::OptionalType: {
      if (input.is_null())
        return graph->insertConstant(IValue());
      else
        return createValue(type->cast<OptionalType>()->getElementType(), input,
                           graph);
    } break;

    case TypeKind::ScalarTypeType: {
      auto dtype = strToDtype.at(input.at("dtype").get<std::string>());
      return graph->insertConstant(dtype);
    } break;

    case TypeKind::DeviceObjType: {
      return graph->insertConstant(IValue(c10::kCUDA));
    } break;

    default: {
      TORCH_CHECK(false, "Type ", *type, " not supported.");
    }
  }
}

static Value *createNode(const json &inputCase, const FunctionSchema &schema,
                         std::shared_ptr<Graph> graph) {
  // Get symbol
  auto symbol = Symbol::fromQualString(schema.name());

  // Parse positional arguments
  auto argJsons = inputCase.at(0).get<std::vector<json>>();
  std::vector<NamedValue> argValues;
  for (auto i : c10::irange(argJsons.size())) {
    auto &input = argJsons[i];
    auto type = schema.arguments()[i].type();
    argValues.push_back(createValue(type, input, graph.get()));
  }

  // Parse keyword arguments
  auto kwargJsons =
      inputCase.at(1).get<std::unordered_map<std::string, json>>();
  std::vector<NamedValue> kwargValues;
  for (auto &pair : kwargJsons) {
    auto argIdx = *schema.argumentIndexWithName(pair.first);
    auto type = schema.arguments()[argIdx].type();
    kwargValues.emplace_back(pair.first,
                             createValue(type, pair.second, graph.get()));
  }

  // Create operation
  auto output = graph->insert(symbol, argValues, kwargValues);
  auto node = output->node();
  if (output->type()->cast<TupleType>())  // fix multi-out operations
    output = graph->createTupleUnpack(output)->insertAfter(node)->output(0);

  return output;
}

static std::unordered_set<Symbol> includedSymbols{
    prim::ListConstruct, prim::TupleUnpack, prim::TupleConstruct};

static void createFusedFunctor(const std::shared_ptr<Graph> &graph) {
  // Infer dtype and device
  ValueTypeMap refinedTypes;
  InferDtypeAndDevice(graph, refinedTypes);

  // Include `ListConstruct` in the graph
  auto tail = graph->return_node(), head = tail->prev();
  for (auto node = head->prev(); node != graph->param_node();
       node = node->prev()) {
    if (includedSymbols.count(node->kind())) {
      node->moveBefore(head);
      head = node;
    }
  }

  // Create fusion group
  commitFusion(head, tail, graph.get(), refinedTypes);

  // Create fusion functor
  FusedOpToParallelization(graph, refinedTypes);
  MapFunctorToParallelization(graph, refinedTypes);
}

static auto rng = at::make_generator<at::CUDAGeneratorImpl>();

static IValue generateInput(TypePtr type) {
  switch (type->kind()) {
    case TypeKind::TensorType: {
      auto tensorTy = type->cast<TensorType>();
      auto shape = *tensorTy->sizes().concrete_sizes();
      auto dtype = *tensorTy->scalarType();
      switch (*tensorTy->scalarType()) {
        case c10::kFloat:
          return at::rand(shape, rng, c10::kFloat, c10::kStrided, c10::kCUDA,
                          c10::nullopt);

        case c10::kLong:
          return at::randint(0, 5, shape, rng, c10::kLong, c10::kStrided,
                             c10::kCUDA, c10::nullopt);

        case c10::kBool:
          return at::randint(0, 2, shape, rng, c10::kBool, c10::kStrided,
                             c10::kCUDA, c10::nullopt);

        default:
          TORCH_CHECK(false, "Dtype ", dtype, " not supported");
      }

    } break;

    default: {
      TORCH_CHECK(false, "Cannot generate input for type ", *type);
    }
  }
}

static void runCase(const json &inputCase, const FunctionSchema &schema) {
  // Construct reference graph
  auto refGraph = std::make_shared<Graph>();
  refGraph->registerOutput(createNode(inputCase, schema, refGraph));
  if (!refGraph->inputs().empty()) ConstantPropagation(refGraph);
  LowerSimpleTuples(refGraph);
  LONG_TAIL_LOG_INFO("Reference graph:");
  LONG_TAIL_LOG_INFO(refGraph->toString());

  // Construct graph with fused functor
  auto compiledGraph = refGraph->copy();
  createFusedFunctor(compiledGraph);
  LONG_TAIL_LOG_INFO("Compiled graph:");
  LONG_TAIL_LOG_INFO(compiledGraph->toString());

  // Generate inputs
  std::vector<IValue> inputs;
  for (auto value : refGraph->inputs())
    inputs.push_back(generateInput(value->type()));

  // Run reference graph
  at::Tensor refOut;
  {
    Code code(refGraph, "test");
    auto stack = inputs;
    InterpreterState(code).run(stack);
    refOut = stack.front().toTensor().cuda();
  }

  // Run compiled graph
  at::Tensor compiledOut;
  {
    Code code(compiledGraph, "test");
    auto stack = inputs;
    InterpreterState(code).run(stack);
    compiledOut = stack.front().toTensor();
  }

  // Compare result
  if (at::allclose(refOut, compiledOut, 1e-3, 1e-5)) return;

  // Report inconsistency
  std::stringstream ss;
  print(ss, "\n", *refGraph);
  for (auto i : c10::irange(inputs.size()))
    print(ss, "\nInput ", i, ": \n", inputs[i], '\n');
  print(ss, "\nReference output: \n", refOut, '\n');
  print(ss, "\nCompiled graph output: \n", compiledOut, '\n');
  TORCH_CHECK(false, ss.str());
}

static void runOpSuite(const json &opSuite) {
  // Find operator from schema
  auto schema = parseSchema(opSuite.at("schema"));
  auto opName = schema.operator_name();
  auto op = findOperatorFor(schema.operator_name());
  TORCH_CHECK(op, "Operator not found for ", opName);

  // Run each test case
  auto inputCases = opSuite.at("cases").get<std::vector<json>>();
  for (auto &testCase : inputCases) runCase(testCase, schema);
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: test_lowering <test-suite-json> <suite-name>...";
    return 1;
  }

  // Load test suite from file
  at::globalContext().lazyInitCUDA();
  std::ifstream suiteFile(argv[1]);
  auto suite = json::parse(suiteFile);

  // Run test suite
  for (auto i = 2u; i < argc; i++) {
    auto opSuite = suite.at(argv[i]);
    runOpSuite(opSuite);
  }

  return 0;
}
