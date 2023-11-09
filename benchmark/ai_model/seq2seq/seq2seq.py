import torch
import functs

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import os

device = torch.device('cuda')
n_warmup = 100
n_run = 100

from functs.utils.evaluate import Timer

MAX_LENGTH = 50
OUTPUT_SIZE = 3797
HIDDEN_SIZE = 256

class LSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        # self.weight_ih_l0 = nn.Parameter(torch.randn(3 * hidden_size, input_size, dtype=torch.float32))
        # self.weight_hh_l0 = nn.Parameter(torch.randn(3 * hidden_size, input_size, dtype=torch.float32))
        # self.bias_ih_l0 = nn.Parameter(torch.randn(3 * hidden_size, dtype=torch.float32))
        # self.bias_hh_l0 = nn.Parameter(torch.randn(3 * hidden_size, dtype=torch.float32))
        self.weight_ih_l0_t = nn.Parameter(torch.randn(4, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh_l0_t = nn.Parameter(torch.randn(4, input_size, hidden_size, dtype=torch.float32))
        # self.bias_ih_l0_t = nn.Parameter(torch.randn(3, 1, hidden_size, dtype=torch.float32))
        # self.bias_hh_l0_t = nn.Parameter(torch.randn(3, 1, hidden_size, dtype=torch.float32))
        self.bias_ih_0 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_0 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_1 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_1 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_2 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_2 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_3 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_3 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        self.input_size = input_size
        nn.init.xavier_uniform_(self.weight_ih_l0_t)
        nn.init.xavier_uniform_(self.weight_hh_l0_t)
    
    def forward(self, x, h, c):
        ih = torch.matmul(x, self.weight_ih_l0_t)
        hh = torch.matmul(h, self.weight_hh_l0_t)
        ih0 = ih[0] + self.bias_ih_0
        hh0 = hh[0] + self.bias_hh_0
        ih1 = ih[1] + self.bias_ih_1
        hh1 = hh[1] + self.bias_hh_1
        ih2 = ih[2] + self.bias_ih_2
        hh2 = hh[2] + self.bias_hh_2
        ih3 = ih[3] + self.bias_ih_3
        hh3 = hh[3] + self.bias_hh_3

        ingate = torch.sigmoid(ih0 + hh0)
        forgetgate = torch.sigmoid(ih1 + hh1)
        cellgate = torch.tanh(ih2 + hh2)
        outgate = torch.sigmoid(ih3 + hh3)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        return h, c

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.gru = LSTMCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.EOS_token = nn.Parameter(torch.full((1,), 0, dtype=torch.long, device=device), requires_grad=False)
        self.EOS_token = 0
        self.SOS_token = 1

    def forward(self, encoder_output, std, h, c):
        # hidden: (1, bs, hidden_size)
        # encoder_outputs: (max_length, bs, hidden_size)
        batch_size = encoder_output.size()[1]
        output_all = torch.zeros(self.max_length, batch_size, dtype=torch.int64, device='cuda') + 0 # Hack for bug in ScatterND on Constant
        output = torch.full((batch_size,), self.SOS_token, dtype=torch.int64, device='cuda')
        cond = True
        # when disable cf
        # id = torch.zeros((), dtype=torch.int64, device='cuda')
        id = 0
        while cond:
            x = self.embedding(output)
            # h = torch.reshape(h, (batch_size, self.hidden_size))
            # lstm start
            ih = torch.matmul(x, self.gru.weight_ih_l0_t)
            hh = torch.matmul(h, self.gru.weight_hh_l0_t)
            ih0 = ih[0] + self.gru.bias_ih_0
            hh0 = hh[0] + self.gru.bias_hh_0
            ih1 = ih[1] + self.gru.bias_ih_1
            hh1 = hh[1] + self.gru.bias_hh_1
            ih2 = ih[2] + self.gru.bias_ih_2
            hh2 = hh[2] + self.gru.bias_hh_2
            ih3 = ih[3] + self.gru.bias_ih_3
            hh3 = hh[3] + self.gru.bias_hh_3

            ingate = torch.sigmoid(ih0 + hh0)
            forgetgate = torch.sigmoid(ih1 + hh1)
            cellgate = torch.tanh(ih2 + hh2)
            outgate = torch.sigmoid(ih3 + hh3)

            c = (forgetgate * c) + (ingate * cellgate)
            h = outgate * torch.tanh(c)
            # lstm end
            output = self.out(h) + std[id]
            output = output.argmax(1)
            output_all[id] = output
            id = id + 1
            cond = bool((torch.max(output) > self.EOS_token).item()) & (id < self.max_length) # when testing torchscript
            # cond = (torch.max(output) > self.EOS_token) & (id < self.max_length)
        return output_all, h


def gen_mask_from_sequence(std):
    bs = std.shape[0]
    padded_std = torch.zeros((bs, MAX_LENGTH), dtype=std.dtype, device=device)
    padded_std[:, :std.shape[1]] = std
    mask = torch.zeros(bs, MAX_LENGTH, OUTPUT_SIZE, device=device)
    mask[torch.arange(bs).unsqueeze(1), torch.arange(MAX_LENGTH).unsqueeze(0), padded_std] = 1000000.0
    mask = mask.transpose(0, 1).contiguous().clone()
    return mask


batch_size = 1

model = AttnDecoderRNN(HIDDEN_SIZE, OUTPUT_SIZE, dropout_p=0.1).to(device).eval()

std = []
MAX_LENGTH = 50
h = torch.randn(batch_size, HIDDEN_SIZE, device=device)
c = torch.randn(batch_size, HIDDEN_SIZE, device=device)
sos = torch.full((batch_size,), model.SOS_token, dtype=torch.int64, device='cuda')
for i in range(batch_size):
    l = 10
    lst = list(range(1, l))
    lst.append(0)
    assert(len(lst) <= MAX_LENGTH)
    # pad to MAX_LENGTH
    lst = lst + [0] * (MAX_LENGTH - len(lst))
    std.append(lst)
std = torch.tensor(std, device=device)
mask = gen_mask_from_sequence(std)
encoder_output = torch.randn(MAX_LENGTH, batch_size, HIDDEN_SIZE, device=device)

jit_model = torch.jit.script(model)
functs_model = functs.jit.script(model)


functs.utils.evaluate_func(model, (encoder_output, mask, h, c), "eager", run_duration=2.)
functs.utils.evaluate_func(jit_model, (encoder_output, mask, h, c), "jit", run_duration=2.)
functs.utils.evaluate_func(functs_model, (encoder_output, mask, h, c), "functs", run_duration=2.)



