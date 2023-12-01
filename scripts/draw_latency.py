import torch

import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", 
         "TorchDynamo + Inductor", 
         "TorchScript + nvfuser", 
         "TorchScript + NNC", 
         "TorchScript + TensorSSA + NNC"]
models = ["YOLOV3", "SSD", "YOLACT", "FCOS", "NASRNN", "LSTM", "seq2seq", "Attention"]
data_1660ti = torch.load("latency_1660ti.pt")
data_1660ti = data_1660ti[0] / data_1660ti

# 3090
data_3090 = torch.load("latency_1660ti.pt")
data_3090 = data_3090[0] / data_3090

platforms = ["(a) RTX 1660Ti", "(b) RTX 3090"]

data = {
    "(b) RTX 3090": data_3090,
    "(a) RTX 1660Ti": data_1660ti,
}

patterns = [ "" , 
            "-" , 
            "o" ,
            "." , 
            "*"]
# illustrate bar figure
colors = ["#cbcbcbff", 
          "#e695bfff", 
          "#82c8e1ff", 
          "#8ee4a1ff", 
          "#ffa13ab4",]

bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

# Post-processing
fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained")
max_ylim = -1
for plat, ax in zip(platforms, axes):
    perf = data[plat]
    print(perf.shape)
    print(len(colors))
    for b, (c, h) in enumerate(zip(colors, perf)):
        print(h)
        print(np.arange(len(models)))
        print(models)
        ax.bar(
            begin_pos + b * bar_width + np.arange(len(models)),
            h,
            width=bar_width,
            color=c,
            # edgecolor="black",
        )
    for m in range(len(models)):
        model_perf = perf[:, m]
        outperf = model_perf[-1] / np.max(model_perf[:-1])
        ax.text(
            begin_pos + (len(tools) - 1) * bar_width + m,
            model_perf[-1],
            "{:.2f}x".format(outperf),
            ha="center",
            va="bottom",
        )
    ax.set_xticks(range(len(models)), models, rotation=20)
    ax.set_xlabel(plat, fontsize=13, weight="bold")
    max_ylim = max(max_ylim, float(np.ceil(np.max(perf))))
    plt.setp(axes, ylim=(0, max_ylim))
    # ax.axhline(y = 1.0, 
    #         color = "#cbcbcbff", 
    #         linestyle = ':') 
# fig.legend(labels=tools, ncol=len(tools), loc="outside upper center")
fig.supylabel("Speed Up", weight='bold')

# for i, _ in enumerate(models):
#     m = data.shape[0] - 1 
#     outperf = data[-1, i] / np.min(data[:-1, i])
#     plt.text(
#         0 - begin_pos,
#         data[-1, i],
#         "{:.2f}x".format(1 / outperf),
#         ha="center",
#         va="bottom",
#         weight='bold'
#     )
fig.legend(labels=tools, ncol=np.ceil(len(tools) / 2), loc="outside upper center", fontsize=16)
# fig.supylabel("Speed Up")
plt.savefig('latency.pdf')
plt.savefig('latency.jpg')
plt.show()





