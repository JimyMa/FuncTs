#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/jit/python/script_init.h>
#include <torch/csrc/utils/pybind.h>

#include <caffe2/serialize/versions.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/compatibility/backport.h>
#include <torch/csrc/jit/mobile/compatibility/model_compatibility.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/quantization.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_sugared_value.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/graph_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_dict.h>
#include <torch/csrc/jit/python/python_list.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/logging.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/testing/hooks_for_testing.h>

#include <torch/csrc/api/include/torch/ordered_dict.h>

#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/csrc/jit/mobile/train/export_data.h>
#include <chrono>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "functs/csrc/jit/api/aot_graph_impl.h"
#include "functs/csrc/jit/api/aot_compilation_unit.h"


namespace torch {
namespace jit {

void initJITFuncTsModuleBindings(PyObject *module) {

  auto m = py::handle(module).cast<py::module>();

  m.def(
      "_create_function_from_graph",
      [](const std::string& qualname, std::shared_ptr<Graph> graph) {
      // TODO this should go in the global Python CU
      auto cu = std::make_shared<AotCompilationUnit>();
      c10::QualifiedName name(qualname);
      auto fn = cu->create_function(std::move(name), std::move(graph));
      return AotStrongFunctionPtr(std::move(cu), fn);
      });
  
  // auto jit = m.def_submodule("_jit");
  py::class_<AotCompilationUnit, std::shared_ptr<AotCompilationUnit>>(
    m, "AotCompilationUnit")
    // .def(
    //     py::init([](const std::string& lang, const uint32_t _frames_up) {
    //       auto cu = std::make_shared<AotCompilationUnit>();
    //       // if (!lang.empty()) {
    //       //   pyCompilationUnitDefine(*cu, lang, nullptr, _frames_up);
    //       // }
    //       // return cu;
    //     }),
    //     py::arg("lang") = "",
    //     py::arg("_frames_up") = 0)

    .def(
        "find_function",
        [](std::shared_ptr<AotCompilationUnit> self, const std::string& name) {
          auto fn = self->find_function(QualifiedName(name));
          if (fn) {
            return c10::optional<AotStrongFunctionPtr>(
                AotStrongFunctionPtr(std::move(self), fn));
          } else {
            return c10::optional<AotStrongFunctionPtr>(c10::nullopt);
          }
        })
    .def(
        "__getattr__",
        [](std::shared_ptr<AotCompilationUnit> self, const std::string& name) {
          auto fn = self->find_function(QualifiedName(name));
          if (fn) {
            return AotStrongFunctionPtr(std::move(self), fn);
          } else {
            throw AttributeError(
                "'AotCompilationUnit' has no attribute '%s'", name.c_str());
          }
        })
    .def(
        "get_functions",
        [](const std::shared_ptr<AotCompilationUnit>& self) {
          auto raw_functions = self->get_functions();
          std::vector<AotStrongFunctionPtr> functions;
          functions.reserve(raw_functions.size());
          for (auto fn : raw_functions) {
            if (fn) {
              functions.emplace_back(self, fn);
            }
          }
          return functions;
        })
    .def("set_optimized", &AotCompilationUnit::set_optimized)
  //   .def(
  //       "define",
  //       pyAotCompilationUnitDefine,
  //       py::arg("src"),
  //       py::arg("rcb") = nullptr,
  //       py::arg("_frames_up") = 0)
    .def(
        "create_function",
        [](std::shared_ptr<AotCompilationUnit>& self,
           const std::string& qualified_name,
           std::shared_ptr<Graph> graph,
           bool should_mangle) {
          Function* fn = self->create_function(
              qualified_name, std::move(graph), should_mangle);
          return AotStrongFunctionPtr(std::move(self), fn);
        },
        py::arg("qualified_name"),
        py::arg("graph"),
        py::arg("should_mangle") = false)
    .def(
        "get_interface",
        [](const std::shared_ptr<AotCompilationUnit>& self,
           const std::string& name) { return self->get_interface(name); })
    .def(
        "get_class",
        [](const std::shared_ptr<AotCompilationUnit>& self,
           const std::string& name) { return self->get_class(name); })
    .def(
        "drop_all_functions",
        [](const std::shared_ptr<AotCompilationUnit>& self) {
          self->drop_all_functions();
        });

  py::class_<AotStrongFunctionPtr>(m, "ScriptFunction", py::dynamic_attr())
      .def(
          "__call__",
          [](py::args args, py::kwargs kwargs) {
            HANDLE_TH_ERRORS
            // see: [pybind11 varargs]
            auto strongPtr = py::cast<AotStrongFunctionPtr>(args[0]);
            Function& callee = *strongPtr.function_;
            py::object result = invokeScriptFunctionFromPython(
                callee,
                // NOLINTNEXTLINE(performance-move-const-arg)
                tuple_slice(std::move(args), 1),
                // NOLINTNEXTLINE(performance-move-const-arg)
                std::move(kwargs));
            return result;
            END_HANDLE_TH_ERRORS_PYBIND
          })
    //   .def(
    //       "save",
    //       [](const AotStrongFunctionPtr& self,
    //          const std::string& filename,
    //          const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
    //         Module module("__torch__.PlaceholderModule");
    //         // [issue 27343]
    //         // Modules have 'training' attributes by default, but due to
    //         // https://github.com/pytorch/pytorch/issues/27343, functions end
    //         // up having a training attribute when they are loaded. This adds
    //         // a fake 'training' attribute that shouldn't be used, but prevents
    //         // jitter on saving and loading. Once that issue is fixed this can
    //         // be deleted.
    //         module.register_attribute("training", BoolType::get(), true);
    //         addFunctionToModule(module, self);
    //         module.save(filename, _extra_files);
    //       },
    //       py::arg("filename"),
    //       py::arg("_extra_files") = ExtraFilesMap())
    //   .def(
    //       "save_to_buffer",
    //       [](const AotStrongFunctionPtr& self,
    //          const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
    //         std::ostringstream buf;
    //         Module module("__torch__.PlaceholderModule");
    //         // see [issue 27343]
    //         module.register_attribute("training", BoolType::get(), true);
    //         addFunctionToModule(module, self);
    //         module.save(buf, _extra_files);
    //         return py::bytes(buf.str());
    //       },
    //       py::arg("_extra_files") = ExtraFilesMap())
      .def_property_readonly(
          "graph",
          [](const AotStrongFunctionPtr& self) {
            return toAotGraphFunction(*self.function_).graph();
          })
      .def_property_readonly(
          "inlined_graph",
          [](const AotStrongFunctionPtr& self) {
            auto g = toAotGraphFunction(*self.function_).graph()->copy();
            Inline(*g);
            return g;
          })
      .def_property_readonly(
          "schema",
          [](const AotStrongFunctionPtr& self) {
            return self.function_->getSchema();
          })
      .def_property_readonly(
          "code",
          [](const AotStrongFunctionPtr& self) {
            std::vector<at::IValue> constants;
            PrintDepsTable deps;

            PythonPrint pp(constants, deps);
            pp.printFunction(*self.function_);
            return pp.str();
          })
      .def(
          "get_debug_state",
          [](const AotStrongFunctionPtr& self) {
            return toAotGraphFunction(*self.function_)
                .get_executor()
                .getDebugState();
          })
      // .def(
      //     "_debug_flush_compilation_cache",
      //     [](const AotStrongFunctionPtr& self) {
      //       toAotGraphFunction(*self.function_)
      //           .get_executor()
      //           .debugFlushCompilationCache();
      //     })
      .def_property_readonly(
          "name",
          [](const AotStrongFunctionPtr& self) { return self.function_->name(); })
      .def(
          "_set_ignore_amp",
          [](AotStrongFunctionPtr& self, bool ignore) {
            auto fn = self.function_;
            TORCH_INTERNAL_ASSERT(fn->isGraphFunction());
            AotGraphFunction& g_fn = toAotGraphFunction(*fn);
            g_fn._set_ignore_amp(ignore);
          })
      .def_property_readonly(
          "qualified_name",
          [](const AotStrongFunctionPtr& self) {
            return self.function_->qualname().qualifiedName();
          })
      .def_property_readonly("__doc__", [](const AotStrongFunctionPtr& self) {
        return self.function_->doc_string();
      });
}
} // namespace jit
} // namespace torch
