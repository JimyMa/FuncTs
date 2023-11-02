from copy import deepcopy

import torch

import functs._C

def script(fn, backend="ts_jit"):
    """ 
    convert PyTorch Program to Ts Graph IR
    backend ["ts_jit", "fait"]: 
    """
    TS_JIT = "ts_jit"
    FAIT = "fait"
    BACKEND_LIST = [TS_JIT, FAIT]

    jit_fn = torch.jit.script(fn.cuda().eval())

    if backend == TS_JIT:
        jit_fn = torch.jit.freeze(jit_fn)
    elif backend == FAIT:
        functs._C._jit_pass_freeze(torch.jit.script(fn).cuda().eval()._c)
    else:
        raise AttributeError("No backend named {}".format(backend))
    
    g = jit_fn.graph
    # functs pass
    torch._C._jit_pass_inline(g)
    functs._C._jit_pass_convert_to_tensorssa(g)
    functs._C._jit_pass_tensorssa_remove_update(g)
    torch._C._jit_pass_dce(g)
    torch._C._jit_pass_cse(g)

    return jit_fn
