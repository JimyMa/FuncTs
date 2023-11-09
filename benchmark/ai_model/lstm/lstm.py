# reference: https://github.com/fawazsammani/mogrifier-lstm-pytorch/blob/master/mog_lstm.py
# changed to be compiled by torchscript

from typing import Tuple
import functs
import torch
import torch.nn as nn
from time import time
import os
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--platform', type=str)
parser.add_argument('--overhead_test', action='store_true')
parser.add_argument('--unroll', dest='unroll', action='store_true')
parser.add_argument('--fix', dest='unroll', action='store_false')
parser.set_defaults(unroll=False)
arguments = parser.parse_args()
platform = arguments.platform

import sys
sys.path.append('../../ast_analyzer/utils')
from functs.utils.timer import Timer
# from nvprof import profile_start, profile_stop, enable_profile
# enable_profile(platform)

n_warmup = 100
n_run = 100
cuda_device = torch.device("cuda:0")
weight_prefix = "../../data/lstm"

def load_tensor_bin(path, dtype=np.float32, device='cuda'):
    with open(f"{path}.shape", "r") as f:
        st = f.readline().strip()
        shape = [int(x) for x in st.split(" ")]
    np_tensor = np.fromfile(f"{path}.bin", dtype=dtype)
    np_tensor = np_tensor.reshape(shape)
    pt_tensor = torch.from_numpy(np_tensor)
    return pt_tensor


class LSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.weight_ih_l0 = nn.Parameter(torch.randn(4 * hidden_size, input_size, dtype=torch.float32))
        self.weight_hh_l0 = nn.Parameter(torch.randn(4 * hidden_size, input_size, dtype=torch.float32))
        self.bias_ih_l0 = nn.Parameter(torch.randn(4 * hidden_size, dtype=torch.float32))
        self.bias_hh_l0 = nn.Parameter(torch.randn(4 * hidden_size, dtype=torch.float32))
        self.weight_ih_l0_t = nn.Parameter(torch.empty(4, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh_l0_t = nn.Parameter(torch.empty(4, input_size, hidden_size, dtype=torch.float32))
        self.bias_ih_l0_t = nn.Parameter(torch.empty(4, 1, hidden_size, dtype=torch.float32))
        self.bias_hh_l0_t = nn.Parameter(torch.empty(4, 1, hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        self.input_size = input_size
    
    def update_param(self):
        self.state_dict()[f"weight_ih_l0_t"][:] = torch.transpose(self.weight_ih_l0.view(4, self.hidden_size, self.input_size), 1, 2)
        self.state_dict()[f"bias_ih_l0_t"][:] = self.bias_ih_l0.reshape((4, 1, self.hidden_size))
        self.state_dict()[f"weight_hh_l0_t"][:] = torch.transpose(self.weight_hh_l0.view(4, self.hidden_size, self.input_size), 1, 2)
        self.state_dict()[f"bias_hh_l0_t"][:] = self.bias_hh_l0.reshape((4, 1, self.hidden_size))

    def forward(self, x, h, c):
        ih = torch.matmul(x, self.weight_ih_l0_t) + self.bias_ih_l0_t
        hh = torch.matmul(h, self.weight_hh_l0_t) + self.bias_hh_l0_t
        # ih0, ih1, ih2, ih3 = torch.split(ih, (1, 1, 1, 1), dim=0)
        ih0 = ih[0]
        ih1 = ih[1]
        ih2 = ih[2]
        ih3 = ih[3]
        # hh0, hh1, hh2, hh3 = torch.split(hh, (1, 1, 1, 1), dim=0)
        hh0 = hh[0]
        hh1 = hh[1]
        hh2 = hh[2]
        hh3 = hh[3]
        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        state_c = (forgetgate * c) + (ingate * cellgate)
        state_h = outgate * torch.tanh(state_c)

        return state_h.clone(), state_h.clone(), state_c.clone()


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(LSTMCell(hidden_size, hidden_size))
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, inputs):  # seq_len, batch, input_size
        batch_size = inputs.shape[1]
        # state_c = [torch.zeros(batch_size, self.hidden_size, device='cuda') for _ in range(10)] # hardcode for ts compile
        state_c = torch.zeros(10,1,  batch_size, self.hidden_size, device='cuda')
        # state_h = [torch.zeros(batch_size, self.hidden_size, device='cuda') for _ in range(10)]
        state_h = torch.zeros(10, 1, batch_size, self.hidden_size, device='cuda')
        for i in range(inputs.size()[0]):
            cur_input = inputs[i]
            for j, layer in enumerate(self.layers):
                c = state_c[j]
                h = state_h[j]
                _, h, c = layer(cur_input, h, c)

                state_c[j] = c
                state_h[j] = h

                cur_input = h

        return state_h[self.num_layers - 1]

class LSTMWrapper(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cuda_device = cuda_device
        self.load_weight()
    
    def load_weight(self):
        for i in range(self.num_layers):
            weight_ih = load_tensor_bin(os.path.join(weight_prefix, f"weight_ih_l{i}"))
            weight_hh = load_tensor_bin(os.path.join(weight_prefix, f"weight_hh_l{i}"))
            bias_ih_0 = load_tensor_bin(os.path.join(weight_prefix, f"bias_ih_0_l{i}"))
            bias_ih_1 = load_tensor_bin(os.path.join(weight_prefix, f"bias_ih_1_l{i}"))
            bias_ih_2 = load_tensor_bin(os.path.join(weight_prefix, f"bias_ih_2_l{i}"))
            bias_ih_3 = load_tensor_bin(os.path.join(weight_prefix, f"bias_ih_3_l{i}"))
            bias_hh_0 = load_tensor_bin(os.path.join(weight_prefix, f"bias_hh_0_l{i}"))
            bias_hh_1 = load_tensor_bin(os.path.join(weight_prefix, f"bias_hh_1_l{i}"))
            bias_hh_2 = load_tensor_bin(os.path.join(weight_prefix, f"bias_hh_2_l{i}"))
            bias_hh_3 = load_tensor_bin(os.path.join(weight_prefix, f"bias_hh_3_l{i}"))
            self.state_dict()[f"lstm.weight_ih_l{i}"][:] = weight_ih.transpose(1, 2).reshape(self.hidden_size * 4, self.input_size)
            self.state_dict()[f"lstm.weight_hh_l{i}"][:] = weight_hh.transpose(1, 2).reshape(self.hidden_size * 4, self.hidden_size)
            self.state_dict()[f"lstm.bias_ih_l{i}"][:] = torch.cat((bias_ih_0, bias_ih_1, bias_ih_2, bias_ih_3))
            self.state_dict()[f"lstm.bias_hh_l{i}"][:] = torch.cat((bias_hh_0, bias_hh_1, bias_hh_2, bias_hh_3))

    def forward(self, input):
        batch_size = input.shape[1]
        state_h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.cuda_device)
        state_c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.cuda_device)
    
        output, (state_h, state_c) = self.lstm(input, (state_h, state_c))
        return output[-1]


def test_model(enable_torch, batch_size, impl, *params):
    input_size, hidden_size, num_layers, seq_len = params
    # model = LSTM(input_size, hidden_size, output_size).to(cuda_device)
    if impl == 'cudnn':
        model = LSTMWrapper(input_size, hidden_size, num_layers).to(cuda_device)
    elif impl == 'loop':
        model = LSTM(input_size, hidden_size, num_layers).to(cuda_device)
    else: raise NotImplementedError
    model.eval()
    if enable_torch == "jit":
        model = torch.jit.script(model)
    elif enable_torch == "functs":
        model = functs.jit.script(model)
    else:
        model = model

    # inp = torch.randn([seq_len, batch_size, input_size], device=cuda_device)
    # inp = load_tensor_bin(os.path.join(weight_prefix, f"inputs_b{batch_size}")).to(cuda_device)
    inp = torch.randn([seq_len, batch_size, input_size], device=cuda_device)
    print("----batch_size={}---torchscript={}----".format(batch_size, enable_torch))
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        _ = model(inp)
        torch.cuda.synchronize()
        print("Time {} ms".format((time() - t0) * 1000))

    if enable_torch == "functs":
        print(model.graph_for(inp))

    timer = Timer()
    torch.cuda.profiler.start()
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        _ = model(inp)
        torch.cuda.synchronize()
        timer.log()
    torch.cuda.profiler.stop()
    timer.report()


def test_train(enable_torch, batch_size, *params):
    input_size, hidden_size, num_layers, seq_len = params
    model = LSTMWrapper(input_size, hidden_size, num_layers).to(cuda_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if enable_torch:
        model = torch.jit.script(model)
    inp = torch.randn([seq_len, batch_size, input_size], device=cuda_device)
    state_h = torch.randn(1, batch_size, hidden_size, device=cuda_device)
    state_c = torch.randn(1, batch_size, hidden_size, device=cuda_device)
    print("----batch_size={}---torchscript={}----".format(batch_size, enable_torch))
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time()
        output = model(inp)
        s = torch.sum(output)
        s.backward()
        torch.cuda.synchronize()
        print("Time {} ms".format((time() - t0) * 1000))

    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        timer.start()
        output = model(inp)
        torch.cuda.synchronize()
        timer.log()
    timer.report()


def export_model(batch_size, input_size, hidden_size, num_layers, seq_len):
    model = LSTMWrapper(input_size, hidden_size, num_layers).to(cuda_device)
    model.eval()
    model = torch.jit.script(model)
    inp = torch.randn([seq_len, batch_size, input_size], device=cuda_device)
    out = model(inp)
    torch.onnx.export(model, (inp), f'lstm.b{batch_size}.onnx', verbose=True, example_outputs=out)


if __name__ == '__main__':
    input_size = 256
    hidden_size = 256
    num_layers = 10
    seq_len = 64


    with torch.no_grad():
        # export_model(1, input_size, hidden_size, output_size, seq_len)
        # export_model(64, input_size, hidden_size, output_size, seq_len)
        # if not arguments.overhead_test:
        #     test_model(True, arguments.bs, 'cudnn', input_size, hidden_size, num_layers, seq_len)
        # else:
        #     if arguments.unroll:
        #         test_model(True, 1, 'unroll', input_size, hidden_size, num_layers, seq_len)
        #     else:

        test_model("jit", 1, 'loop', input_size, hidden_size, num_layers, seq_len)
        test_model(False, 1, 'loop', input_size, hidden_size, num_layers, seq_len)
        test_model("functs", 1, 'loop', input_size, hidden_size, num_layers, seq_len)

        # test_model(False, 1, 'cudnn', input_size, hidden_size, num_layers, seq_len)
        # test_model(True, 1, 'cudnn', input_size, hidden_size, num_layers, seq_len)
        # test_model(False, 64, 'cudnn', input_size, hidden_size, num_layers, seq_len)
        # test_model(True, 64, 'cudnn', input_size, hidden_size, num_layers, seq_len)

        # test_model(True, 1, 'loop', input_size, hidden_size, num_layers, seq_len)
        # test_model(True, 1, 'unroll', input_size, hidden_size, num_layers, seq_len)




