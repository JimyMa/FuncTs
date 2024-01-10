#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include <functs/csrc/parallel/ops/homo_conv.h>
#include <functs/csrc/utils/logging.h>
#include <cstddef>

using namespace c10::functs_parallel;

namespace torch {
namespace jit {

class HomoConvBuilder {
 public:
  HomoConvBuilder(Node* node) {
    attribute_extraction(node);
    compile();
  }
  void run(Stack& stack) {}

 private:
  bool with_bias;
  size_t parallel_level;
  size_t in_channels;
  size_t out_channels;
  std::tuple<size_t, size_t> kernel_size;
  std::tuple<size_t, size_t> stride;
  std::tuple<size_t, size_t> padding;
  std::tuple<size_t, size_t> dilation;
  size_t groups;
  void attribute_extraction(Node* node) {
    // extract functs_parallel::HomoConv

    // parallel_level
    auto parallel_level_value = node->input(0);
    parallel_level = toIValue(parallel_level_value).value().toInt();

    // inputs
    // auto conv_ins = node->input(1);

    // weight
    auto weight_value = node->input(2);
    auto weight_shape =
        weight_value->type()->cast<c10::TensorType>()->symbolic_sizes();
    LONG_TAIL_ASSERT(weight_shape.isComplete(), "weight shape must be static");
    out_channels = weight_shape.at(0).value();
    in_channels = weight_shape.at(1).value();
    auto kernel_h = weight_shape.at(2).value();
    auto kernel_w = weight_shape.at(3).value();
    kernel_size = {kernel_h, kernel_w};

    // bias
    auto bias_value = node->input(3);
    with_bias = (prim::Constant == bias_value->node()->kind() &&
                 toIValue(bias_value)->isNone())
        ? false
        : true;

    // stride
    auto stride_value = node->input(4);
    auto stride_h = toIValue(stride_value)->toIntList()[0];
    auto stride_w = toIValue(stride_value)->toIntList()[1];
    stride = {stride_h, stride_w};

    // pads
    auto pad_value = node->input(5);
    auto pad_h = toIValue(pad_value)->toIntList()[0];
    auto pad_w = toIValue(pad_value)->toIntList()[1];
    padding = {pad_h, pad_w};

    // dilation
    auto dilation_value = node->input(6);
    auto dilation_h = toIValue(dilation_value)->toIntList()[0];
    auto dilation_w = toIValue(dilation_value)->toIntList()[1];
    dilation = {dilation_h, dilation_w};

    // group
    groups = toIValue(node->input(7))->toInt();

    LONG_TAIL_LOG_INFO(
        "HomoConv extraction from functs_parallel::homo_conv node done"
        << "\nwith_bias: " << with_bias << "\nin_channels: " << in_channels
        << "\nout_channels: " << out_channels << "\nkernel_size: "
        << std::get<0>(kernel_size) << ", " << std::get<1>(kernel_size) << "\n"
        << "\nstride: " << std::get<0>(stride) << ", " << std::get<1>(stride)
        << "\npadding: " << std::get<0>(padding) << ", " << std::get<1>(padding)
        << "\ndilation: " << std::get<0>(dilation) << ", "
        << std::get<1>(dilation) << "\n")
  }
  void compile() {}
};

static Operation CreateTeOperator(const Node* node) {
  auto homo_conv_builder = std::make_shared<HomoConvBuilder>(node);
  return [homo_conv_builder](Stack& stack) -> int {
    homo_conv_builder->run(stack);
    return 0;
  };
}

RegisterOperators HomoConvOps({
    torch::jit::Operator(
        c10::functs_parallel::HomoConv,
        CreateTeOperator,
        AliasAnalysisKind::PURE_FUNCTION),
});

} // namespace jit
} // namespace torch
