from _ast import AnnAssign
import ast
from time import time
from typing import Union

import logging

import torch
import torch.nn as nn

from collections import OrderedDict, defaultdict

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--bs', type=int, default=1)
# arguments = parser.parse_args()

cuda_device = torch.device("cuda:0")
n_warmup = 100
n_run = 100


class NasRNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size):
        super(NasRNN, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(8, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(torch.randn(8, hidden_size, hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def forward(self, inputs):  # seq_len, batch, input_size
        state_c = torch.zeros(self.batch_size, self.hidden_size, device='cuda')
        state_m = torch.zeros(self.batch_size, self.hidden_size, device='cuda') # TODO: batch_size from shape
        inputs_bcast = inputs.unsqueeze(1)
        ihs = torch.matmul(inputs_bcast, self.weight_ih)
        for i in range(inputs.size()[0]): # change to 1000 for fully unrolled exp
            inp = inputs[i]
            state_m = torch.reshape(state_m, (self.batch_size, self.hidden_size))

            ih = ihs[i]
            hh = torch.matmul(state_m, self.weight_hh)

            i0 = ih[0]
            i1 = ih[1]
            i2 = ih[2]
            i3 = ih[3]
            i4 = ih[4]
            i5 = ih[5]
            i6 = ih[6]
            i7 = ih[7]

            h0 = hh[0]
            h1 = hh[1]
            h2 = hh[2]
            h3 = hh[3]
            h4 = hh[4]
            h5 = hh[5]
            h6 = hh[6]
            h7 = hh[7]

            layer1_0 = torch.sigmoid(i0 + h0)
            layer1_1 = torch.relu(i1 + h1)
            layer1_2 = torch.sigmoid(i2 + h2)
            layer1_3 = torch.relu(i3 + h3)
            layer1_4 = torch.tanh(i4 + h4)
            layer1_5 = torch.sigmoid(i5 + h5)
            layer1_6 = torch.tanh(i6 + h6)
            layer1_7 = torch.sigmoid(i7 + h7)

            l2_0 = torch.tanh(layer1_0 * layer1_1)
            l2_1 = torch.tanh(layer1_2 + layer1_3)
            l2_2 = torch.tanh(layer1_4 * layer1_5)
            l2_3 = torch.sigmoid(layer1_6 + layer1_7)

            # Inject the cell
            l2_0_v2 = torch.tanh(l2_0 + state_c)

            # Third layer
            state_c = l2_0_v2 * l2_1
            l3_1 = torch.tanh(l2_2 + l2_3)

            # Final layer
            state_m = torch.tanh(state_c * l3_1)

        state = state_c + state_m
        return state


# import tsd

INPUT_SIZE = 256
HIDDEN_SIZE = 256
SEQ_LEN = 1000

nasrnn = NasRNN(1, INPUT_SIZE, HIDDEN_SIZE).cuda().eval()
nasrnn_jit_fn = torch.jit.script(nasrnn)
print(nasrnn_jit_fn.graph)

import functs
nasrnn_functs_fn = functs.jit.script(nasrnn)
print(nasrnn_functs_fn.graph)


a = torch.rand([SEQ_LEN, 1, INPUT_SIZE]).cuda().float()

import time
# warm up
for _ in range(10):
    nasrnn_jit_fn(a)
    nasrnn_functs_fn(a)

# warm up
for _ in range(10):
    nasrnn_jit_fn(a)
    nasrnn_functs_fn(a)

begin = time.time()
for _ in range(100):
    nasrnn_jit_fn(a)
mid = time.time()
for _ in range(100):
    nasrnn_functs_fn(a)
end = time.time()

print("functs: ", end - mid)
print("jit: ", mid - begin)




