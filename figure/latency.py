import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", "TorchScript", "FuncTs"]

models = ["yolov3_postprocess", "yolov3", "ssd_postprocess", "ssd", "yolact_postprocess", "yolact", "nasrnn", "lstm", "seq2seq", "attention", "normalize"]
latency_eager = 1 / np.array([1.182, 1.182+1.54, 2.869, 2.869 + 1.35, 4.48, 4.48 + 3.41, 276.7, 92.58, 3.173, 5.883, 0.7049])
latency_jit = 1 / np.array([0.899, 0.899+1.54, 2.079, 2.079 + 1.35, 3.18, 3.18 + 3.41, 112.9, 36.59, 1.896, 4.089, 0.5708])
# latency_tracing_jit = [1.101, 2.858]
latency_functs = 1 / np.array([0.442, 0.442+1.54, 0.960, 0.960 + 1.35, 1.11, 1.11+3.41, 38.43, 18.48, 1.013, 2.409, 0.1612])

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

fig = plt.figure(figsize=(12, 3), layout='constrained')
for b, (c, h) in enumerate(zip(colors, data)):
    plt.bar(
        begin_pos + b * bar_width + np.arange(len(models)),
        h,
        width=bar_width,
        color=c,
        edgecolor="black",
        label="Normalized Performance"
    )
plt.xticks(range(len(models)), models, rotation=10)
plt.ylabel("Normalized Performance")
fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center")

plt.savefig('latency.pdf')
plt.savefig('latency.jpg')
plt.show()