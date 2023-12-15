import functs._C
import torch

import torch._C._te as te

from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase


class TestImmutTensorExpr(TestCase):
    def test_immut_select(self) -> None:
        g_string = """
graph(%self : Float(10, 20, strides=[20, 1], device=cuda:0),
      %index : int):
      %dim : int = prim::Constant[value=0]()
      %res : Float(20, strides=[1], device=cuda:0) = immut::select(%self, %dim, %index)
      return (%res)
"""
        g = torch.parse_ir(g_string)
        nnc_module = te.TensorExprKernel(g)
        args = (torch.rand(10, 20).float().cuda(), 0)
        torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
        return

    def test_immut_select_rev(self) -> None:
        g_string = """
graph(%self : Float(3000, 4, strides=[4, 1], device=cuda:0),
      %src : Float(3000, strides=[1], device=cuda:0),
      %idx : int):
      %dim : int = prim::Constant[value=1]()
      %res : Float(3000, 4, strides=[4, 1], device=cuda:0) = immut::select_rev(%self, %src, %dim, %idx)
      return (%res)
"""
        g = torch.parse_ir(g_string)
        nnc_module = te.TensorExprKernel(g)
        args = (torch.rand(3000, 4).float().cuda(), torch.rand(3000).float().cuda(), 0)
        torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
        return

    def test_immut_slice(self) -> None:
        g_string = """
graph(%self : Float(3000, 4, strides=[4, 1], device=cuda:0),
      # 0
      %start : int,
      # 3
      %end : int,
      # 4
      %step : int):
      %dim : int = prim::Constant[value=1]()
      %res : Float(3000, 1, strides=[1, 1], device=cuda:0) = immut::slice(%self, %dim, %start, %end, %step)
      return (%res)
"""
        g = torch.parse_ir(g_string)
        nnc_module = te.TensorExprKernel(g)
        args = (torch.rand(3000, 4).float().cuda(), 0, 3, 4)
        torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
        return

    def test_immut_slice_rev(self) -> None:
        g_string = """
graph(%self : Float(3000, 4, strides=[4, 1], device=cuda:0),
      %src : Float(3000, 1, strides=[1, 1], device=cuda:0),
      # 0
      %start : int,
      # 3
      %end : int,
      # 4
      %step : int):
      %dim : int = prim::Constant[value=1]()
      %res : Float(3000, 4, strides=[4, 1], device=cuda:0) = immut::slice_rev(%self, %src, %dim, %start, %end, %step)
      return (%res)
"""
        g = torch.parse_ir(g_string)
        nnc_module = te.TensorExprKernel(g)
        args = (
            torch.rand(3000, 4).float().cuda(),
            torch.rand(3000, 1).float().cuda(),
            0,
            3,
            4,
        )
        torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
        return

    def test_immut_unsqueeze(self) -> None:
        g_string = """
graph(%self : Float(3000, 4, strides=[4, 1], device=cuda:0)):
      %dim : int = prim::Constant[value=0]()
      %res : Float(1, 3000, 4, strides=[12000, 4, 1], device=cuda:0) = immut::unsqueeze(%self, %dim)
      return (%res)
"""
        g = torch.parse_ir(g_string)
        nnc_module = te.TensorExprKernel(g)
        args = (torch.rand(3000, 4).float().cuda(),)
        torch.testing.assert_close(nnc_module.run(args), nnc_module.fallback(args))
        return


if __name__ == "__main__":
    run_tests()
