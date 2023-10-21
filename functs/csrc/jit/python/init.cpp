#include <torch/csrc/utils/pybind.h>

#include <functs/csrc/jit/passes/convert_to_tensorssa.h>
#include <functs/csrc/jit/python/init.h>

namespace torch {
namespace jit {

void initJITFuncBindings(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();
  auto jit = m.def_submodule("_jit");
  m.def("_jit_pass_convert_to_tensorssa", ConvertToTensorSSA);
}
} // namespace jit
} // namespace torch
