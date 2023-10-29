import torch
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)

from torch.testing import FileCheck

import torch._C._te as te
import functs._C


class TestImmutTensorExpr(TestCase):
  def test_immut_select(self) -> None:
    g_string = """
graph(%self : Float(10, 20, strides=[20, 1], device=cuda),
      %index : int):
      %dim : int = prim::Constant[value=0]()
      %res : Float(20, strides=[1], device=cuda) = immut::select(%self, %dim, %index)
      return (%res)
"""
    g = torch.parse_ir(g_string)
    nnc_module = te.TensorExprKernel(g)
    args = (torch.rand(10, 20).float().cuda(), 0)
    torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
    return

  def test_immut_slice(self) -> None:
    g_string = """
graph(%self : Float(3000, 4, strides=[4, 1], device=cuda),
      # 0
      %start : int,
      # 3
      %end : int,
      # 4
      %step : int):
      %dim : int = prim::Constant[value=1]()
      %res : Float(3000, 1, strides=[1, 1], device=cuda) = immut::slice(%self, %dim, %start, %end, %step)
      return (%res)
"""
    g = torch.parse_ir(g_string)
    nnc_module = te.TensorExprKernel(g)
    args = (torch.rand(3000, 4).float().cuda(), 0, 3, 4)
    torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
    return
  
  def test_immut_slice_rev(self) -> None:
    g_string = """
graph(%self : Float(3000, 4, strides=[4, 1], device=cuda),
      %src : Float(3000, 1, strides=[1, 1], device=cuda),
      # 0
      %start : int,
      # 3
      %end : int,
      # 4
      %step : int):
      %dim : int = prim::Constant[value=1]()
      %res : Float(3000, 4, strides=[4, 1], device=cuda) = immut::slice_rev(%self, %src, %dim, %start, %end, %step)
      return (%res)
"""
    g = torch.parse_ir(g_string)
    nnc_module = te.TensorExprKernel(g)
    args = (torch.rand(3000, 4).float().cuda(), torch.rand(3000, 1).float().cuda(), 0, 3, 4)
    torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
    return
  
  def test_immut_unsqueeze(self) -> None:
    g_string = """
graph(%self : Float(3000, 4, strides=[4, 1], device=cuda)):
      %dim : int = prim::Constant[value=0]()
      %res : Float(1, 3000, 4, strides=[12000, 4, 1], device=cuda) = immut::unsqueeze(%self, %dim)
      return (%res)
"""
    g = torch.parse_ir(g_string)
    nnc_module = te.TensorExprKernel(g)
    args = (torch.rand(3000, 4).float().cuda(), )
    torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
    return


if __name__ == "__main__":
  run_tests()



