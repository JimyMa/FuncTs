#include <c10/core/impl/PyInterpreter.h>
#include <memory>

#include <Python.h>
#include <functs/csrc/jit/passes/convert_to_tensorssa.h>
#include <functs/csrc/jit/passes/fait/fait_pipeline.h>
#include <functs/csrc/jit/passes/remove_inplace.h>
#include <functs/csrc/jit/passes/shape_analysis.h>
#include <functs/csrc/jit/python/init.h>
#include <passes/freeze_module.h>

// #include <torch/csrc/Dtype.h>
// #include <torch/csrc/jit/api/function_impl.h>
// #include <torch/csrc/jit/python/python_arg_flatten.h>
// #include <torch/csrc/jit/python/python_custom_class.h>
// #include <torch/csrc/jit/python/python_ir.h>
// #include <torch/csrc/jit/python/python_tracer.h>
// #include <torch/csrc/jit/python/python_tree_views.h>
// #include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace jit {

void initJITFuncTsBindings(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();
  auto jit = m.def_submodule("_jit");
  m.def("_jit_pass_dumb_remove_inter_precedure_mutation",
        DumbRemoveInterPrecedureMutation);
  m.def("_jit_pass_remove_inplace", RemoveInplace);
  m.def("_jit_pass_convert_to_tensorssa", ConvertToTensorSSA);
  m.def("_jit_pass_tensorssa_remove_update", TensorSSARemoveUpdate);
  m.def("_jit_pass_fait_pipeline", FaitPipeline);
  m.def("_jit_pass_fait_shape_infer", FaitGetRefineType);
  m.def("_jit_pass_freeze", Freeze);
  m.def("_jit_pass_clone", Clone);
  m.def("_jit_get_code", [](std::shared_ptr<Graph> g) -> Code {
    return Code(g, "<func on demand>");
  });
  m.def("_jit_run_code", [](Code code, const py::tuple &inputs) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    Stack stack;
    stack.reserve(inputs.size()); // captures?
    for (auto& obj : inputs) {
      stack.push_back(toTypeInferredIValue(obj));
    }
    InterpreterState(code).run(stack);
    auto return_stack = createPyObjectForStack(std::move(stack));;
    PyGILState_Release(gstate);
    return return_stack;
  });
}
} // namespace jit
} // namespace torch
