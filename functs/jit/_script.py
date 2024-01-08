import inspect
from copy import deepcopy
from typing import List

import torch

import functs._C


class AotScriptFunction(object):
    """
    Interface of *Aot Graph Runtime*
    """

    def __init__(self, aot_script_fn, **kwargs):
        self._aot_script_fn = aot_script_fn

    @property
    def graph(self):
        return self._aot_script_fn.graph

    def __call__(self, *args, **kwargs):
        return self._aot_script_fn("forward", *args, **kwargs)


def extract_type_hint_from_tensor(input_):
    if isinstance(input_, bool):
        return torch.BoolType.get()
    elif isinstance(input_, int):
        return torch.IntType.get()
    elif isinstance(input_, float):
        return torch.FloatType.get()
    elif isinstance(input_, torch.Tensor):
        return (
            torch.TensorType.get()
            .with_dtype(input_.dtype)
            .with_sizes(input_.shape)
            .with_device(input_.device)
        )
    elif isinstance(input_, list) or isinstance(input_, tuple):
        return torch.TupleType([extract_type_hint_from_tensor(elem) for elem in input_])
    else:
        raise TypeError(
            "unsupported type {} when build aot graph at the type hint stage"
        )


def shape_infer(fn, example_input, refined_types={}) -> None:
    type_hint = [extract_type_hint_from_tensor(input_) for input_ in example_input]
    g = fn.graph
    functs._C._jit_pass_fait_shape_infer(g, type_hint, refined_types)


def build(
    fn: torch.jit._script.ScriptModule, example_input: List[object]
) -> AotScriptFunction:
    """
    compile functionalized model to aot_graph
    """
    if not isinstance(fn, torch.jit._script.ScriptModule):
        raise AttributeError("{} only functionalized jit" "ScriptModule can be built")

    type_hint = [extract_type_hint_from_tensor(input_) for input_ in example_input]

    g = fn.graph
    functs._C._jit_pass_fait_pipeline(g, type_hint)
    aot_script_fn = functs._C._create_function_from_graph("forward", g)
    return AotScriptFunction(aot_script_fn=aot_script_fn)


def script(
    fn: torch.jit._script.ScriptModule,
    backend="jit",
    remove_update=True,
    enable_dce_cse=True,
    add_clone=False,
) -> torch.jit._script.ScriptModule:
    """
    convert PyTorch Program to Ts Graph IR and perform functionalization
    backend ["ts_jit", "fait"]:
    """
    JIT = "jit"
    AOT = "aot"
    # BACKEND_LIST = [JIT, AOT]

    jit_fn = torch.jit.script(fn)
    if not inspect.isfunction(fn):
        jit_fn = jit_fn.cuda().eval()
        if backend == JIT:
            jit_fn = torch.jit.freeze(jit_fn)
        elif backend == AOT:
            functs._C._jit_pass_freeze(torch.jit.script(fn).cuda().eval()._c)
        else:
            raise AttributeError("No backend named {}".format(backend))

    g = jit_fn.graph
    if add_clone:
        functs._C._jit_pass_dumb_remove_inter_precedure_mutation(g)
    # functs pass
    torch._C._jit_pass_inline(g)
    functs._C._jit_pass_convert_to_tensorssa(g)
    if remove_update:
        functs._C._jit_pass_tensorssa_remove_update(g)
    if enable_dce_cse:
        torch._C._jit_pass_dce(g)
        torch._C._jit_pass_cse(g)
        torch._C._jit_pass_constant_propagation(g)
    return jit_fn
