#include <vector>
#include <Python.h>

#include <functs/csrc/jit/python/init.h>

extern "C" PyObject* initModule();

PyObject* module;

static std::vector<PyMethodDef> methods;

PyObject* initModule() {
  static struct PyModuleDef functsmodule = {
      PyModuleDef_HEAD_INIT, "functs._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&functsmodule);

  torch::jit::initJITFuncBindings(module);

  return module;
}


