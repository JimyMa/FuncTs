import functs
import argparse
import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity


parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--maxlength', type=int, default=50)
parser.add_argument('--tool', type=str, default="all")

START_LEN = 32
NUM_HEAD = 12
SIZE_PER_HEAD = 64


class Attention(nn.Module):
    def __init__(self, num_head, size_per_head, seq_len):
        super().__init__()
        self.weight_q = nn.Parameter(torch.randn(
            num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_k = nn.Parameter(torch.randn(
            num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_v = nn.Parameter(torch.randn(
            num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_o = nn.Parameter(torch.randn(
            num_head, size_per_head, size_per_head, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight_q)
        nn.init.xavier_uniform_(self.weight_k)
        nn.init.xavier_uniform_(self.weight_v)
        nn.init.xavier_uniform_(self.weight_o)
        self.num_head = num_head
        self.size_per_head = size_per_head
        self.start_len = START_LEN
        self.seq_len = seq_len

    def forward(self, x, k, v):  # (batch_size, num_head, 1, size_per_head)
        k = k + 0.0
        v = v + 0.0
        batch_size = x.size()[0]
        gen_id = self.start_len
        attn = torch.zeros(batch_size, self.num_head, 1,
                           self.seq_len, device='cuda')
        for i in range(k.size()[2] - self.start_len):
            q = torch.matmul(x, self.weight_q)
            k[:, :, gen_id, :] = torch.reshape(torch.matmul(
                x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
            v[:, :, gen_id, :] = torch.reshape(torch.matmul(
                x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
            attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
            attn = attn * 0.125
            attn = torch.softmax(attn, dim=3)
            x = torch.matmul(attn, v)
            x = torch.matmul(x, self.weight_o)
            gen_id = gen_id + 1
        return k.clone(), v.clone(), x.clone()


if __name__ == "__main__":
    arguments = parser.parse_args()
    SEQ_LEN = arguments.maxlength
    batch_size = arguments.bs
    x = torch.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD).cuda()
    k = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD,
                    dtype=torch.float32, device='cuda')
    k[:, :, :START_LEN, :] = torch.randn(
        batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
    v = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD,
                    dtype=torch.float32, device='cuda')
    v[:, :, :START_LEN, :] = torch.randn(
        batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')

    model = Attention(NUM_HEAD, SIZE_PER_HEAD, SEQ_LEN).cuda().eval()
    jit_model = torch.jit.script(model)
    dynamo_model = torch.compile(model, dynamic=True)
    nvfuser_model = torch.jit.freeze(torch.jit.script(model))

    functs_model = functs.jit.script(model)
    fait_model = functs.jit.build(functs.jit.script(
        torch.jit.freeze(torch.jit.script(model))), [x, k, v])

    print(torch.allclose(model(x, k, v)[0], functs_model(x, k, v)[0]))
    print(torch.allclose(model(x, k, v)[1], functs_model(x, k, v)[1]))
    print(torch.allclose(model(x, k, v)[2], functs_model(x, k, v)[2]))
    with torch.no_grad():
        timer_eager = functs.utils.evaluate_func(
            model, [x, k, v], "eager", run_duration=2.)
        timer_jit = functs.utils.evaluate_func(
            jit_model, [x, k, v], "jit", run_duration=2.)
        timer_functs = functs.utils.evaluate_func(
            functs_model, [x, k, v], "functs", run_duration=2.)
        # timer_fait = functs.utils.evaluate_func(fait_model, [x, k, v], "fait", run_duration=2.)
        timer_dynamo = functs.utils.evaluate_func(
            dynamo_model, [x, k, v], "dynamo", run_duration=2.)

        torch._C._jit_set_nvfuser_enabled(True)
        timer_nvfuser = functs.utils.evaluate_func(
            nvfuser_model, [x, k, v], "nvfuser", run_duration=2.)
        torch._C._jit_set_nvfuser_enabled(False)

        # print(jit_model.graph_for(x, k, v))
        # print(functs_model.graph_for(x, k, v))

        # print("profiler latency cuda graph")
        # for i in range(2, 5 + 2):
        #     print("iter per capture: {}".format(i + 10))
        #     functs.utils.evaluate.evaluate_func(model, [x, k, v], "attention eager", run_duration=2., enable_cudagraph=True, iter_per_capture=i + 10)
        # functs.utils.evaluate.evaluate_func(dynamo_model, [x, k, v], "attention dynamo", run_duration=2., enable_cudagraph=True, iter_per_capture=i + 10)
        # functs.utils.evaluate.evaluate_func(jit_model, [x, k, v], "attention jit", run_duration=2., enable_cudagraph=True, iter_per_capture=i + 10)
        # functs.utils.evaluate.evaluate_func(functs_model, [x, k, v], "attention functs", run_duration=2., enable_cudagraph=True, iter_per_capture=i + 10)

        # print("profiler latency cuda graph")
        # for i in range(1, 5 + 1):
        #     torch._C._jit_set_nvfuser_enabled(True)
        #     functs.utils.evaluate.evaluate_func(nvfuser_model, [x, k, v], "attention nvfuser", run_duration=2., enable_cudagraph=True, iter_per_capture=i + 10)
        #     torch._C._jit_set_nvfuser_enabled(False)
        # print(functs.utils.proifler_func(model, (x, k, v), "attention eager", run_duration=2.0).key_metrics)
        # print(functs.utils.proifler_func(jit_model, (x, k, v), "attention jit", run_duration=2.0).key_metrics)
        # print(functs.utils.proifler_func(functs_model, (x, k, v), "attention functs", run_duration=2.0).key_metrics)
        # timer_fait = functs.utils.proifler_func(torch._C._jit_run_code, [code, ("", x, k, v)], "fait", run_duration=2., export_json="fait.json")

        # if arguments.tool in ["all", "eager"]:
        #     print(functs.utils.proifler_func(
        #         model, [x, k, v], "eager", run_duration=1.0, export_json="eager").key_metrics)

        # if arguments.tool in ["all", "jit"]:
        #     print(functs.utils.proifler_func(jit_model, [
        #           x, k, v], "jit", run_duration=1.0, export_json="jit").key_metrics)
        # if arguments.tool in ["all", "dynamo"]:
        #     print(functs.utils.proifler_func(dynamo_model, [
        #           x, k, v], "dynamo", run_duration=1.0, export_json="dynamo").key_metrics)
        # if arguments.tool in ["all", "functs"]:
        #     print(functs.utils.proifler_func(functs_model, [
        #           x, k, v], "functs", run_duration=1.0, export_json="functs").key_metrics)

        # if arguments.tool in ["all", "nvfuser"]:
        #     torch._C._jit_set_nvfuser_enabled(True)
        #     print(functs.utils.evaluate.proifler_func(nvfuser_model, [
        #           x, k, v], "nvfuser", run_duration=1.0, export_json="nvfuser").key_metrics)
        #     torch._C._jit_set_nvfuser_enabled(False)
