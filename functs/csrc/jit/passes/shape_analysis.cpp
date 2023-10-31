// #include <ATen/core/dynamic_type.h>
// #include <ATen/core/jit_type.h>
// #include <c10/macros/Export.h>
// #include <functs/csrc/jit/passes/shape_analysis.h>
// #include <functs/csrc/jit/utils/refine_type.h>
// #include <functs/csrc/jit/utils/type_utils.h>
// #include <memory>
// #include <torch/csrc/jit/ir/ir.h>
// #include <torch/csrc/jit/passes/shape_analysis.h>
// #include <unordered_map>
// #include <unordered_set>

// namespace torch {
// namespace jit {

// using ValueTypeMap = std::unordered_map<Value *, TypePtr>;

// #define INFER_PARAMS Node *node, ValueTypeMap &refinedTypes

// static c10::optional<int64_t>
// refineDimSizeIndex(Value *indexValue,
//                    const c10::optional<int64_t> &defaultIfNone) {
//   c10::optional<int64_t> index;
//   auto ival = toIValue(indexValue);
//   if (!ival)
//     return c10::nullopt;
//   if (ival->isNone())
//     index = defaultIfNone;
//   else if (ival->isInt())
//     index = ival->toInt();
//   return index;
// }

// static OperatorSet sameShapeOps{
//     "immut::access(Tensor src) -> Tensor",
//     "immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor",
//     "immut::select_rev(Tensor self, Tensor src, int dim, int index) ->
//     Tensor", "immut::slice_rev(Tensor self, Tensor src, int dim=0, SymInt?
//     start=None, " "SymInt? end=None, SymInt step=1) -> Tensor"};

// static c10::SymbolicShape passSameShape(INFER_PARAMS) {
//   return node->input(0)->type()->cast<TensorType>()->symbolic_sizes();
// }

// static OperatorSet immutSelectOp{
//     "immut::select(Tensor src, int dim, int index) -> Tensor",
// };

// static c10::SymbolicShape inferShapeImmutSelectOp(INFER_PARAMS) {
//   // Process argument
//   auto inShape = getShape(node->input(0)->type());
//   if (!inShape)
//     return {};
//   auto rank = inShape->size();
//   auto dimIVal = toIValue(node->input(1));
//   if (!dimIVal)
//     return getRankedShape(rank - 1);
//   auto dim = dimIVal->toInt();
//   if (dim < 0)
//     dim += rank;

//   // Infer output shape
//   inShape->erase(inShape->begin() + dim);
//   return *inShape;
// }

// static OperatorSet immutSliceOp{
//     "immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
//     "end=None, SymInt step=1) -> Tensor",
// };

// static c10::SymbolicShape inferShapeImmutSliceOp(INFER_PARAMS) {
//   // Process argument
//   auto inShape = getShape(node->input(0)->type());
//   if (!inShape)
//     return {};
//   auto rank = inShape->size();
//   auto dimIVal = toIValue(node->input(1));
//   if (!dimIVal)
//     return getRankedShape(rank);
//   auto dim = dimIVal->toInt();
//   if (dim < 0)
//     dim += rank;

//   // Process dimension range
//   auto dimSize = inShape->at(dim);
//   auto start = refineDimSizeIndex(node->input(2), 0);
//   auto end = refineDimSizeIndex(node->input(3), dimSize);
//   auto step = refineDimSizeIndex(node->input(4), 1);
//   auto outDimSize = tryApply<int64_t>(
//       [](int64_t dimSize, int64_t start, int64_t end, int64_t step) {
//         if (start < 0)
//           start += dimSize;
//         if (end < 0)
//           end += dimSize;
//         return (std::min(end, dimSize) - start - 1) / step + 1;
//       },
//       dimSize, start, end, step);

//   // Compute output shape
//   ShapeVec outShape;
//   for (auto i : c10::irange(rank)) {
//     ShapeDim size;
//     if (i == dim)
//       size = outDimSize;
//     else
//       size = inShape->at(i);
//     outShape.push_back(size);
//   }

//   return outShape;
// }

// static std::initializer_list<
//     std::pair<OperatorSet, c10::SymbolicShape (*)(INFER_PARAMS)>>
//     shapeFuncInit{
//         {sameShapeOps, passSameShape},
//         {immutSelectOp, inferShapeImmutSelectOp},
//         {immutSliceOp, inferShapeImmutSliceOp},
//     };

// static bool initialized = false;
// OperatorMap<c10::SymbolicShape (*)(INFER_PARAMS)> shapeFuncs;
// OperatorMap<c10::ScalarType (*)(INFER_PARAMS)> dtypeFuncs;

// void initTensorshapeFuncs() {
//   if (initialized)
//     return;
//   for (auto &pair : shapeFuncInit)
//     shapeFuncs.insert(pair.first, pair.second);
// }

// bool PropagateImmutInputShapesBlock(
//     Block *block, std::unordered_set<Node *> &shapeAnalysisDone) {
//   bool any_changed = false;
//   auto nodes = block->nodes();
//   for (auto node : nodes) {
//     for (auto &b : node->blocks()) {
//       any_changed |= PropagateImmutInputShapesBlock(b, shapeAnalysisDone);
//     }
//     auto op = node->maybeOperator();
//     if (!op)
//       continue;
//     if (shapeFuncs.contains(*op)) {
//       auto changed = !shapeAnalysisDone.count(node);
//       if (changed) {
//         node->dump();
//         std::cout
//             << node->output()->type()->cast<c10::TensorType>()->isComplete()
//             << std::endl;
//         std::cout << node->output()
//                          ->type()
//                          ->cast<c10::TensorType>()
//                          ->symbolic_sizes()
//                          .isComplete()
//                   << std::endl;
//         std::cout
//             <<
//             node->output()->type()->cast<c10::TensorType>()->annotation_str()
//             << std::endl;
//         std::cout << node->output()
//                          ->type()
//                          ->cast<c10::TensorType>()
//                          ->symbolic_sizes()
//                          .rank()
//                   << std::endl;
//         std::cout << node->output()
//                          ->type()
//                          ->cast<c10::TensorType>()
//                          ->symbolic_sizes()
//                          .sizes()
//                          ->size()
//                   << std::endl;
//         std::cout << node->output()
//                          ->type()
//                          ->cast<c10::TensorType>()
//                          ->symbolic_sizes()
//                          .sizes()
//                          ->empty()
//                   << std::endl;
//         std::unordered_map<Value *, TypePtr> refinedTypes;
//         auto shape = (*shapeFuncs.find(*op))(node, refinedTypes);
//         auto output = node->output();
//         auto input = node->input(0);
//         auto input_scalar_type =
//             input->type()->cast<TensorType>()->scalarType();
//         auto input_device = input->type()->cast<TensorType>()->device();

//         node->output()->setType(
//             output->type()->cast<TensorType>()->withScalarType(
//                 input_scalar_type));
//         node->output()->setType(
//             output->type()->cast<TensorType>()->withSymbolicShapes(shape));
//         node->output()->setType(
//             output->type()->cast<TensorType>()->withDevice(input_device));
//         shapeAnalysisDone.insert(node);
//       }
//       any_changed = any_changed | changed;
//     }
//   }
//   return any_changed;
// }

// TORCH_API bool
// PropagateImmutInputShapes(const std::shared_ptr<Graph> &graph,
//                           std::unordered_set<Node *> &shapeAnalysisDone) {
//   initTensorshapeFuncs();
//   auto block = graph->block();
//   return PropagateImmutInputShapesBlock(block, shapeAnalysisDone);
// }

// TORCH_API
// void PropagateInputShapesImmutDLC(const std::shared_ptr<Graph> &graph) {

//   std::function<void(Block *)> initShapeAnalysis =
//       [&initShapeAnalysis](Block *b) -> void {
//     auto nodes = b->nodes();
//     for (auto node : nodes) {
//       auto blocks = node->blocks();
//       for (auto block : blocks) {
//         initShapeAnalysis(block);
//       }
//       for (auto output : node->outputs()) {
//         if (output->type()->cast<c10::TensorType>()) {
//           output->setType(
//               output->type()->cast<c10::TensorType>()->withSymbolicShapes({}));
//         }
//       }
//     }
//   };

//   initShapeAnalysis(graph->block());

//   bool any_changed = true;
//   std::unordered_set<Node *> shapeAnalysisDone;
//   while (any_changed) {
//     PropagateInputShapes(graph);
//     any_changed = PropagateImmutInputShapes(graph, shapeAnalysisDone);
//   }
// }

// } // namespace jit
// } // namespace torch
