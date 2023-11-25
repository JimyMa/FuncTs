import typing

import torch
import functs


class Func(torch.nn.Module):
    def forward(self, l: typing.List[torch.Tensor]):
        x = l[0]
        y = l[1]
        z = l[2]

        return x + y + z


jit_func = functs.jit.script(Func())
type_hint = [torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([20, 20]).with_device(torch.device("cuda"))])]

functs._C._jit_pass_fait_pipeline(jit_func.graph, type_hint)

c = functs._C._jit_get_code(jit_func.graph)

qq = torch.save([torch.randn([1, 20]).float().cuda(), torch.ones([20, 20]).float().cuda(), torch.ones([20, 20]).float().cuda()], "qq.pt")

qq_data = torch.load("qq.pt")

print(functs._C._jit_run_code(c, ("", qq_data, )))

print(jit_func.graph)




