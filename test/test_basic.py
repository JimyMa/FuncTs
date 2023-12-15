import functs._C
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def branch_loop(a: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    tmp_0 = d.clone()
    tmp_1 = tmp_0[1]
    if a.nonzero().sum():
        tmp_0 = tmp_0 + 1
        for i in range(10):
            tmp_1.copy_(tmp_0[i + 1])
    else:
        tmp_0 = tmp_0 + tmp_1
        tmp_1.copy_(tmp_0[2])
    return tmp_1 + tmp_0


def dataflow(a: torch.Tensor, b: torch.Tensor):
    a[0].copy_(b[0])
    a[1].copy_(b[1])
    a[0:1] = b[0:1]
    return a + a[0:1]


def loop(a: torch.Tensor, b: torch.Tensor):
    for i in range(10):
        a[i].copy_(b[i + 1])
    return a


def branch(a: torch.Tensor, b: torch.Tensor, cond: int):
    tmp_1 = a[1]
    if cond > 0:
        tmp_1.copy_(b[0])
    else:
        tmp_1.copy_(b[1] * 2)
    return a


class TestBasic(TestCase):
    def gen_functs_fn(self, fn) -> torch.ScriptFunction:
        jit_fn = torch.jit.script(fn)
        g = jit_fn.graph
        functs._C._jit_pass_convert_to_tensorssa(g)
        functs._C._jit_pass_tensorssa_remove_update(g)
        torch._C._jit_pass_dce(g)
        return jit_fn

    def run_func_test(self, fn, *args):
        jit_fn = self.gen_functs_fn(fn)
        self.assertTrue(torch.allclose(jit_fn(*args), fn(*args)))

    def test_dataflow(self) -> None:
        a = torch.rand([1024, 1024])
        b = torch.rand([1024, 1024])
        self.run_func_test(dataflow, a, b)

    def test_loop(self) -> None:
        a = torch.rand([1024, 1024])
        b = torch.rand([1024, 1024])
        self.run_func_test(loop, a, b)

    def test_branch(self) -> None:
        a = torch.rand([1024, 1024])
        b = torch.rand([1024, 1024])
        cond = torch.randn(1)
        self.run_func_test(branch, a, b, cond)

    def test_branch_loop(self) -> None:
        a = torch.rand([1024, 1024])
        b = torch.rand([1024, 1024])
        c = torch.rand([1024, 1024])
        self.run_func_test(branch_loop, a, b, c)


if __name__ == "__main__":
    run_tests()
