import torch
from torch import nn


START_LEN = 32
SEQ_LEN = 64
NUM_HEAD = 12
SIZE_PER_HEAD = 64

class Attention(nn.Module):
    def __init__(self, num_head, size_per_head):
        super().__init__()
        self.weight_q = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_k = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_v = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        self.weight_o = nn.Parameter(torch.randn(num_head, size_per_head, size_per_head, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight_q)
        nn.init.xavier_uniform_(self.weight_k)
        nn.init.xavier_uniform_(self.weight_v)
        nn.init.xavier_uniform_(self.weight_o)
        self.num_head = num_head
        self.size_per_head = size_per_head
        self.start_len = START_LEN
        self.seq_len = SEQ_LEN

    def forward(self, x, k, v): # (batch_size, num_head, 1, size_per_head)
        k = k + 0.0
        v = v + 0.0
        batch_size = x.size()[0]
        gen_id = self.start_len
        attn = torch.zeros(batch_size, self.num_head, 1, self.seq_len, device='cuda')
        for i in range(k.size()[2] - self.start_len):
            q = torch.matmul(x, self.weight_q)
            k[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_k), (batch_size, self.num_head, self.size_per_head))
            v[:, :, gen_id, :] = torch.reshape(torch.matmul(x, self.weight_v), (batch_size, self.num_head, self.size_per_head))
            attn = torch.matmul(k, q.transpose(2, 3)).transpose(2, 3)
            attn = attn * 0.125
            attn = torch.softmax(attn, dim=3)
            x = torch.matmul(attn, v)
            x = torch.matmul(x, self.weight_o)
            gen_id = gen_id + 1
        return k.clone(), v.clone(), x.clone()
    
batch_size = 1

model = Attention(NUM_HEAD, SIZE_PER_HEAD).cuda().eval()
jit_model = torch.jit.script(model)
import functs
functs_model = functs.jit.script(model)

x = torch.randn(batch_size, NUM_HEAD, 1, SIZE_PER_HEAD).cuda()
k = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
k[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
v = torch.zeros(batch_size, NUM_HEAD, SEQ_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')
v[:, :, :START_LEN, :] = torch.randn(batch_size, NUM_HEAD, START_LEN, SIZE_PER_HEAD, dtype=torch.float32, device='cuda')

functs.utils.evaluate_func(model, [x, k, v], "eager", run_duration=2.)
functs.utils.evaluate_func(jit_model, [x, k, v], "jit model", run_duration=2.)
functs.utils.evaluate_func(functs_model, [x, k, v], "functs model", run_duration=2.)



