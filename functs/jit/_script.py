import torch

import functs._C

def script(fn):
    """ extend torchscript script module to support functionalization """
    jit_fn = torch.jit.script(fn)
    g = jit_fn.graph

    # functs pass
    torch._C._jit_pass_inline(g)
    functs._C._jit_pass_convert_to_tensorssa(g)
    functs._C._jit_pass_tensorssa_remove_update(g)
    torch._C._jit_pass_dce(g)
    torch._C._jit_pass_cse(g)

    return jit_fn
