import torch
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)

from torch.testing import FileCheck

import functs._C
  

class TestBasic(TestCase):
  def test_immut_slice(self) -> None:
    def immut_slice(a: torch.Tensor):
        return a + a[:, 0] + a[:, 1]

    fn = torch.jit.script(immut_slice)
    g = fn.graph
    a = torch.randn(10, 10).float().cuda()
    functs._C._jit_pass_convert_to_tensorssa(g)
    functs._C._jit_pass_tensorssa_remove_update(g)
    torch._C._jit_pass_dce(g)
    fn(a)
    g = fn.graph_for(a)
    print(g)
    FileCheck.check("prim::TensorExpr")


if __name__ == "__main__":
  run_tests()



