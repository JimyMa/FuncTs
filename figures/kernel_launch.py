import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch Eager", "TorchDynamo + Inductor",  "TorchScript + nvfuser", "TorchScript + NNC", "FuncTs"]

models = ["YOLOV3_postprocess", "SSD_postprocess", "YOLACT_postprocess", "NASRNN", "LSTM", "seq2seq", "Attention", "Normalize"]
latency_eager = np.array([96, 242, 269 ,1655, 10822, 312, 323, 4])
latency_jit = np.array([83, 236, 231, 154, 3202, 102, 322, 3])
latency_dynamo = np.array([64, 161, 147, 106, 10881, 311, 288, 1])
latency_functs = np.array([34, 103, 92, 104, 1921, 92, 288, 1])
latency_nvfuser = np.array([75, 187, 181, 203, 3200, 92, 321, 2])


# latency_tracing_jit = [1.101, 2.858]


latency_eager_normal = latency_eager / latency_eager * latency_eager
latency_jit_normal = latency_jit / latency_eager * latency_eager
latency_functs_normal = latency_functs / latency_eager * latency_eager

latency_nvfuser_normal = latency_nvfuser / latency_eager * latency_eager
latency_dynamo_normal = latency_dynamo / latency_eager * latency_eager

data = np.stack([latency_eager_normal, latency_dynamo_normal, latency_nvfuser_normal, latency_jit_normal, latency_functs_normal])

# illustrate bar figure
# colors = ["orchid", "cornflowerblue", "salmon",]
colors = ["#003366", "#E31B23", "#005CAB", "#784c22", "#FFC325",]

bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=16)
plt.rc("pdf", fonttype=42)

fig, axes = plt.subplots(4, int(latency_eager_normal.shape[0] / 4), figsize=(10, 9), layout="constrained")
# ax_big = fig.add_subplot(111)  
# fig = plt.figure(figsize=(14, 3), layout='constrained')
for b, (c, h) in enumerate(zip(colors, data)):
    for i, hm in enumerate(h):
        print(axes.shape)
        print(i)
        print(i % 2, i // 2)
        ax = axes[ i // 2, i % 2]
       
        ax.bar(
            begin_pos + b * bar_width + np.arange(1),
            hm,
            width=bar_width,
            color=c,
            # edgecolor="black",
            label="Normalized Performance"
        )
        ax.get_xaxis().set_visible(False)
        ax.set_title(models[i], fontsize=17, weight="bold", y = -0.15)
        if (i+1) % 2:
            ax.set_ylabel("Kernel Launches", fontsize=17, weight="bold")

ax = fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)


fig.legend(labels=tools, ncol=3, loc="outside upper center")

plt.savefig('kernel_launch.pdf')
plt.savefig('kernel_launch.jpg')
plt.show()