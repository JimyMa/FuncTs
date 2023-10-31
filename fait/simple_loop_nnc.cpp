//
// Created by jimyma on 2/3/23.
//

#include <string>

#include "ATen/Context.h"

#include "torch/csrc/jit/tensorexpr/analysis.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"

#include "tensorexpr/functor_parallization.h"
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/types.h>

/*
 * Torch Simple Loop Code
// input shape: [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]
//def forward(self, pred_maps: List[torch.Tensor]):
//    featmap_strides = [32, 16, 8]
//    num_imgs = pred_maps[0].shape[0]
//
//    flatten_preds = []
//    flatten_strides = []
//    for pred, stride in zip(pred_maps, featmap_strides):
//        pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 85)
//        pred[..., :2].sigmoid_()
//        flatten_preds.append(pred)
//        flatten_strides.append(torch.tensor(stride).expand(pred.size(1)))
//    flatten_preds = torch.cat(flatten_preds, dim=1)
//    flatten_strides = torch.cat(flatten_strides)
//
//    return flatten_preds, flatten_strides
*/

using namespace torch::jit::tensorexpr;

bool verbose = true;

int main() {
  // Step 0: Init CUDA Enviroment
  at::globalContext().lazyInitCUDA();

  // Step 1: Define Compute Functor
  // Step 1.0: Define Shape Args
  auto N = LongImm::make(1);
  auto C = LongImm::make(255);
  auto H = VarHandle("dyn_shape_h", kLong);
  auto W = VarHandle("dyn_shape_w", kLong);

  // Step 1.1: Define Input Tensor Args
  BufHandle pred("pred", {N, C, H, W}, kDouble);
  VarHandle stride("stride", kLong);

  // Step 1.2: Define Compute Op
  // permute
  Tensor permute_0 = Compute(
    "permute_0",
    {N, H, W, C},
    [&](const std::vector<VarHandle>& axes) {
      return pred.load(axes[0], axes[2], axes[3], axes[1]);
    });

  // reshape
  auto reshape_dim_0 = LongImm::make(85);
  auto reshape_dim_1 = H * W * C / reshape_dim_0;
  Tensor reshape_0 = Compute(
    "reshape_0",
    {N, reshape_dim_1, reshape_dim_0},
    [&](const std::vector<VarHandle>& axes) {
      auto flatten = axes[0] * reshape_dim_1 * reshape_dim_0 + axes[1] * reshape_dim_0 + axes[2];
      auto dim_c = flatten % C;
      auto res_0 = flatten / C;
      auto dim_w = res_0 % W;
      auto res_1 = res_0 / W;
      auto dim_h = res_1 % H;
      auto dim_n = res_1 / H;
      return permute_0.load(dim_n, dim_h, dim_w, dim_c);
    });

  // sigmoid
  Tensor sigmoid_0 = Compute(
    "sigmoid_0",
    {N, reshape_dim_1, reshape_dim_0},
    [&](const std::vector<VarHandle>& axes) {
      return CompareSelect::make(
              axes[2],
              LongImm::make(2),
              sigmoid(reshape_0.load(axes[0], axes[1], axes[2])),
              reshape_0.load(axes[0], axes[1], axes[2]),
              CompareSelectOperation::kLT);
    });
  // tensor
  Tensor tensor_0 = Compute(
    "tensor_0",
    {reshape_dim_0, },
    [&](const std::vector<VarHandle>& axes) {
      return stride;
    });

  // Step 1.4: Register Output Args
  std::unordered_set<BufPtr> bufOutputs;
  bufOutputs.insert(sigmoid_0.buf());
  bufOutputs.insert(tensor_0.buf());

  // Step 1.5: Construct Statement
  auto block = alloc<Block>(std::vector<StmtPtr>({}));

  block->append_stmt(permute_0.stmt());
  block->append_stmt(reshape_0.stmt());
  block->append_stmt(sigmoid_0.stmt());
  block->append_stmt(tensor_0.stmt());

  // Step 2.0: Loop Schedule
  LoopNest l(block, bufOutputs);
  LoopNest::sanitizeNames(l.root_stmt());

  if (verbose) {
    std::cout << "Original Functor: " << std::endl;
    std::cout << to_string(l.root_stmt()) << std::endl;
  }


  l.simplify();

  if (verbose) {
    std::cout << "after simplify: " << std::endl;
    std::cout << to_string(l.root_stmt()) << std::endl;
  }

  // Step 2.1: Compute Inline
  l.inlineIntermediateBufs(true);
  // l.optimizeConditionals();

  auto stmt_ = l.root_stmt();
  if (verbose) {
    std::cout << "after compute inline: " << std::endl;
    std::cout << to_string(stmt_) << std::endl;
  }

  // Step 2.2: Loop Binding
  for (auto buf : bufOutputs) {
    std::vector<ForPtr> loops = l.getLoopStmtsFor(buf);
    if (loops.empty()) {
      // This happens when Buf is 0-dim
      continue;
    }
    ForPtr flattened = nullptr;
    LoopNest::flatten(loops, &flattened);
    assert(flattened);

    int loopLevels = -1;
    const int kDefaultLoopLevels = 2;

    loopLevels = (loopLevels > 0) ? loopLevels : kDefaultLoopLevels;

    int blockCount = -1;
    int blockSize = -1;

    ForPtr inner;
    const int kDefaultBlockSize = 512;
    blockSize = (blockSize > 0) ? blockSize : kDefaultBlockSize;
    LoopNest::splitWithMask(flattened, blockSize, &inner);
    flattened->set_gpu_block_index(0);
    inner->set_gpu_thread_index(0);
  }


  // Step 3: Functor Parallelization
  // Step 3.1: Add a New Loop
  auto new_loop_axis = VarHandle("new_axis_i", kLong);
  stmt_ = alloc<For>(new_loop_axis.node(),
                     LongImm::make(0).node(),
                     LongImm::make(3).node(),
                     stmt_);
  static_to<For>(stmt_)->set_gpu_block_index(1);

  if (verbose) {
    std::cout << "after loop binding: " << std::endl;
    std::cout << to_string(stmt_) << std::endl;
  }

  // Step 3.2: Arguments Replacement
  int64_t list_size = 3;
  // shapes
  std::vector<VarHandle> H_parall_args;
  std::vector<VarHandle> W_parall_args;

  // inputs
  std::vector<BufHandle> pred_parall_args;

  std::vector<VarHandle> stride_parall_args;
  // outputs
  std::vector<BufHandle> sigmoid_parall_args;
  std::vector<BufHandle> tensor_parall_args;
  sigmoid_parall_args.reserve(list_size);
  tensor_parall_args.reserve(list_size);
  H_parall_args.reserve(list_size);
  H_parall_args.reserve(list_size);
  pred_parall_args.reserve(list_size);
  stride_parall_args.reserve(list_size);

  for (int i = 0; i < list_size; i++) {
    auto H_parall_arg = VarHandle("H_" + std::to_string(i), kLong);
    auto W_parall_arg = VarHandle("w_" + std::to_string(i), kLong);
    H_parall_args.push_back(H_parall_arg);
    W_parall_args.push_back(W_parall_arg);

    pred_parall_args.push_back({"pred_"+ std::to_string(i), {N, C, H_parall_arg, W_parall_arg}, kDouble});
    stride_parall_args.push_back({"stride_" + std::to_string(i), kLong});

    auto sigmoid_shape_parall_dim_1 = H_parall_arg * W_parall_arg * C / LongImm::make(85) / 2;
    auto sigmoid_shape_parall_dim_0 = LongImm::make(85);

    sigmoid_parall_args.push_back({"sigmoid_" + std::to_string(i),
                                   {N, sigmoid_shape_parall_dim_1, sigmoid_shape_parall_dim_0},
                                   kDouble});

    tensor_parall_args.push_back({"tensor_" + std::to_string(i),
                                  {LongImm::make(85)},
                                  kLong});
  }

  stmt_ = FunctorParallization::parallel_functor_load(stmt_, list_size, new_loop_axis.node(),
                                                      {
                                                        {pred.node(), pred_parall_args}
                                                      }, {
                                                        {stride.node(), stride_parall_args}
                                                      });

  stmt_ = FunctorParallization::parallel_functor_store(stmt_, list_size, new_loop_axis.node(),
                                                       {
                                                        {sigmoid_0.buf(), sigmoid_parall_args},
                                                        {tensor_0.buf(), tensor_parall_args}
                                                       });

  stmt_ = FunctorParallization::parallel_functor_shape(stmt_, list_size, new_loop_axis.node(),
                                                       {
                                                         {H.node(), H_parall_args},
                                                         {W.node(), W_parall_args}
                                                       });

  l.prepareForCodegen();
  l.simplify();
  auto stmt = l.root_stmt();
  IRSimplifier::simplify(stmt);
  if (verbose) {
    std::cout << "after loop parallization: " << std::endl;
    std::cout << to_string(stmt_) << std::endl;
  }

  std::vector<CodeGen::BufferArg> bufferArgs_;
  bufferArgs_.insert(bufferArgs_.end(),pred_parall_args.begin(), pred_parall_args.end());
  bufferArgs_.insert(bufferArgs_.end(),stride_parall_args.begin(), stride_parall_args.end());
  bufferArgs_.insert(bufferArgs_.end(),H_parall_args.begin(), H_parall_args.end());
  bufferArgs_.insert(bufferArgs_.end(),W_parall_args.begin(), W_parall_args.end());
  bufferArgs_.insert(bufferArgs_.end(),sigmoid_parall_args.begin(), sigmoid_parall_args.end());
  bufferArgs_.insert(bufferArgs_.end(),tensor_parall_args.begin(), tensor_parall_args.end());

  auto codegen_ = CreateCodeGen(
          "cuda_codegen",
          stmt_,
          bufferArgs_,
          at::kCUDA);

  if (verbose){
    std::cout << "codegen text" << std::endl;
    std::cout << codegen_->getCodeText() << std::endl;
  }

  // Runtime
  auto N_runtime = 1;
  auto C_runtime = 255;

  auto H_0_runtime = 10l;
  auto H_1_runtime = 20l;
  auto H_2_runtime = 40l;

  auto W_0_runtime = 10l;
  auto W_1_runtime = 20l;
  auto W_2_runtime = 40l;

  auto pred_0 = at::ones({N_runtime, C_runtime, H_0_runtime, W_0_runtime}).to(at::kDouble).cuda();
  auto pred_1 = at::ones({N_runtime, C_runtime, H_1_runtime, W_1_runtime}).to(at::kDouble).cuda();
  auto pred_2 = at::ones({N_runtime, C_runtime, H_2_runtime, W_2_runtime}).to(at::kDouble).cuda();

  auto stride_0 = 32l;
  auto stride_1 = 16l;
  auto stride_2 = 8l;

  auto sigmoid_dim_0 = 1;
  auto sigmoid_dim_1_0 = 300;
  auto sigmoid_dim_1_1 = 1200;
  auto sigmoid_dim_1_2 = 4800;
  auto sigmoid_dim_2 = 85;

  auto sigmoid_0_runtime = codegen_->empty_strided(
          {sigmoid_dim_0, sigmoid_dim_1_0, sigmoid_dim_2},
          {sigmoid_dim_2 * sigmoid_dim_1_0, sigmoid_dim_2, 1},
          c10::kDouble,
          c10::kStrided,
          c10::kCUDA,
          false);

  auto sigmoid_1_runtime = codegen_->empty_strided(
          {sigmoid_dim_0, sigmoid_dim_1_1, sigmoid_dim_2},
          {sigmoid_dim_2 * sigmoid_dim_1_1, sigmoid_dim_2, 1},
          c10::kDouble,
          c10::kStrided,
          c10::kCUDA,
          false);

  auto sigmoid_2_runtime = codegen_->empty_strided(
          {sigmoid_dim_0, sigmoid_dim_1_2, sigmoid_dim_2},
          {sigmoid_dim_2 * sigmoid_dim_1_2, sigmoid_dim_2, 1},
          c10::kDouble,
          c10::kStrided,
          c10::kCUDA,
          false);

  auto tensor_0_runtime = codegen_->empty_strided(
          {sigmoid_dim_2, },
          {1, },
          c10::kLong,
          c10::kStrided,
          c10::kCUDA,
          false);

  auto tensor_1_runtime = codegen_->empty_strided(
          {sigmoid_dim_2, },
          {1, },
          c10::kLong,
          c10::kStrided,
          c10::kCUDA,
          false);

  auto tensor_2_runtime = codegen_->empty_strided(
          {sigmoid_dim_2, },
          {1, },
          c10::kLong,
          c10::kStrided,
          c10::kCUDA,
          false);


  std::vector<CodeGen::CallArg> runArgs = {pred_0.data_ptr(), pred_1.data_ptr(), pred_2.data_ptr(),
                                           stride_0, stride_1, stride_2,
                                           H_0_runtime, H_1_runtime, H_2_runtime,
                                           W_0_runtime, W_1_runtime, W_2_runtime,
                                           sigmoid_0_runtime.data_ptr(),
                                           sigmoid_1_runtime.data_ptr(),
                                           sigmoid_2_runtime.data_ptr(),
                                           tensor_0_runtime.data_ptr(),
                                           tensor_1_runtime.data_ptr(),
                                           tensor_2_runtime.data_ptr()};


  codegen_->call(runArgs);

//  std::cout << sigmoid_0_runtime << std::endl;
//  std::cout << sigmoid_1_runtime << std::endl;
//  std::cout << sigmoid_2_runtime << std::endl;
//
//  std::cout << tensor_0_runtime << std::endl;
//  std::cout << tensor_1_runtime << std::endl;
//  std::cout << tensor_2_runtime << std::endl;
//
  return 0;
}
