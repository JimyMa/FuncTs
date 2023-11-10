import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", "TorchScript", "FuncTs"]

models = ["yolov3_postprocess", "ssd_postprocess", "yolact_postprocess", "nasrnn", "lstm", "seq2seq", "attention", "normalize"]
latency_eager = np.array([96, 242, 269 ,33005, 10822, 312, 323, 4])
latency_jit = np.array([83, 236, 231, 3004, 3202, 102, 322, 3])
# latency_tracing_jit = [1.101, 2.858]
latency_functs = np.array([34, 103, 92, 2004, 1921, 92, 289, 1])

latency_eager_normal = latency_eager / latency_eager
latency_jit_normal = latency_jit / latency_eager
latency_functs_normal = latency_functs / latency_eager

data = np.stack([latency_eager_normal, latency_jit_normal, latency_functs_normal])

# illustrate bar figure
colors = ["orchid", "cornflowerblue", "salmon",]

bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

fig = plt.figure(figsize=(11, 3), layout='constrained')
for b, (c, h) in enumerate(zip(colors, data)):
    plt.bar(
        begin_pos + b * bar_width + np.arange(len(models)),
        h,
        width=bar_width,
        color=c,
        # edgecolor="black",
        label="Normalized Performance"
    )
plt.xticks(range(len(models)), models, rotation=10)
plt.ylabel("Normalized Performance")
fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center")

plt.savefig('kernel_launch.pdf')
plt.savefig('kernel_launch.jpg')
plt.show()