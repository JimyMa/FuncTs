#include <Python.h>
#include <functs/csrc/parallel/ops/homo_conv.h>
#include <functs/csrc/parallel/python/ops_init.h>
#include <passes/freeze_module.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace jit {

void initParallelOpsFuncTsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto jit = m.def_submodule("_parallel");
  m.def("invoke_homo_conv", &homo_invoke);
}
} // namespace jit
} // namespace torch
