import time

import torch
import functs

import ssd_bbox


model = ssd_bbox.SSDBBox().cuda().eval()


# torchscript
jit_model = torch.jit.freeze(torch.jit.script(model))

# fait
cls_scores: [torch.Size([1, 486, 20, 20]), torch.Size([1, 486, 10, 10]), torch.Size([1, 486, 5, 5]), torch.Size([1, 486, 3, 3]), torch.Size([1, 486, 2, 2]), torch.Size([1, 486, 1, 1])]
bbox_preds: [torch.Size([1, 24, 20, 20]), torch.Size([1, 24, 10, 10]), torch.Size([1, 24, 5, 5]), torch.Size([1, 24, 3, 3]), torch.Size([1, 24, 2, 2]), torch.Size([1, 24, 1, 1])]
type_hint = [torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 10, 10]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 5, 5]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 3, 3]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 2, 2]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 1, 1]).with_device(torch.device("cuda"))],),
             torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 10, 10]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 5, 5]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 3, 3]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 2, 2]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 1, 1]).with_device(torch.device("cuda"))])]
        
functs_model = functs.jit.script(model)


input = torch.load("ssd_feat.pt")[0]
jit_model(*input)
functs_model(*input)

for i in range(10):
    functs_model(*input)
    jit_model(*input)
    model(*input)

for i in range(10):
    functs_model(*input)
    jit_model(*input)
    model(*input)

begin = time.time()
for i in range(10):
    o_functs = functs_model(*input)
mid_0 = time.time()
# for i in range(1000):
#     code = torch._C._jit_get_code(fait_model.graph)
mid_1 = time.time()
for i in range(10):
    jit_model(*input)
mid_2 = time.time()
for i in range(10):
    model(*input)
end = time.time()

print("functs: ", mid_0 - begin)
# print("fait: ", mid_1 - mid_0)
print("torchscript: ", mid_2 - mid_1)
print("eager: ", end - mid_2)