#include <vector>

#include <functs/csrc/jit/python/init.h>
#include <functs/csrc/jit/python/script_init.h>
#include <functs/csrc/jit/tensorexpr/lowerings.h>
#include <functs/csrc/parallel/python/ops_init.h>

extern "C" PyObject* initModule();

PyObject* module;

static std::vector<PyMethodDef> methods;

PyObject* initModule() {
  static struct PyModuleDef functsmodule = {
      PyModuleDef_HEAD_INIT, "functs._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&functsmodule);

  torch::jit::tensorexpr::init_nnc_ext();
  torch::jit::initJITFuncTsBindings(module);
  torch::jit::initJITFuncTsModuleBindings(module);
  torch::jit::initParallelOpsFuncTsBindings(module);

  return module;
}
