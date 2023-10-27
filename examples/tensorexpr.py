from typing import List

import torch

custom_op_schema_literal = "nnc_custom::add_mul(Tensor a, Tensor b, Tensor c) -> Tensor"

g_string = """
graph(%a : Float(10, 20, strides=[20, 1], device=cpu), 
      %b : Float(10, 20, strides=[20, 1], device=cpu),
      %c : Float(10, 20, strides=[20, 1], device=cpu)):
    %res : Float(10, 20, strides=[20, 1], device=cpu) = nnc_custom::add_mul(%a, %b, %c)
    return (%res)
"""
def computOutput(a: List[int], b: List[int], c: List[int]):
    expandedSizes: List[int] = []
    dimsA = len(a)
    dimsB = len(b)
    dimsC = len(c)
    ndim = max(dimsA, dimsB, dimsC)
    for i in range(ndim):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        dimC = dimsC - 1 - offset
        sizeA = a[dimA] if (dimA >= 0) else 1
        sizeB = b[dimB] if (dimB >= 0) else 1
        sizeC = a[dimC] if (dimC >= 0) else 1

        if sizeA != sizeB and sizeB != sizeC and sizeA != 1 and sizeB != 1 and sizeC != 1:
            # TODO: only assertion error is bound in C++ compilation right now
            raise AssertionError(
                "The size of tensor a {} must match the size of tensor b ("
                "{} and c {}) at non-singleton dimension {}".format(sizeA, sizeB, sizeC, i)
            )

        expandedSizes.append(max(sizeA, sizeB, sizeC))

    return expandedSizes

g = torch.parse_ir(g_string)
g_shape_infer = torch.jit.script(computOutput).graph

cus_schema = torch.parse_schema(custom_op_schema_literal)
torch._C._jit_register_shape_compute_graph_for_schema(cus_schema, g_shape_infer)
torch._C._jit_pass_fuse_tensorexprs(g, 1, False, True)
print(g)

