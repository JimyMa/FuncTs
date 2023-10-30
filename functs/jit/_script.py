from copy import deepcopy

import torch

import functs._C

def script(fn):
    """ 
    extend torchscript script module to support functionalization
    Note: To divide the script function from functs and jit,
    please use torch.nn.Module without python function as your first choice.
    
    >>> func_fn = functs.jit.script(py_fn)
    >>> jit_fn = torch.jit.script(py_fn)
    >>> func_fn == jit_fn
    >>> True

    >>> func_fn = functs.jit.script(NNModule())
    >>> jit_fn = torch.jit.script(NNModule())
    >>> func_fn == jit_fn
    >>> False

    For break the linke of jit_fn and func_fn in the situation of py_fn,
    bynow you should correct the function signature.

    """
    jit_fn = torch.jit.script(fn)
    g = jit_fn.graph

    # functs pass
    torch._C._jit_pass_inline(g)
    functs._C._jit_pass_convert_to_tensorssa(g)
    functs._C._jit_pass_tensorssa_remove_update(g)
    torch._C._jit_pass_dce(g)
    torch._C._jit_pass_cse(g)

    return jit_fn
