import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", "TorchDynamo + Inductor", "TorchScript + nvfuser", "TorchScript + NNC", "TorchScript + TensorSSA + NNC"]

models =                        ["YOLOV3_postprocess", "SSD_postprocess", "YOLACT_postprocess", "NASRNN", "LSTM", "seq2seq", "Attention", "Normalize"]
latency_eager =    1 / np.array([1.443,                 4.092,             4.354,               13.23,    99.93,  19.13,     3.241,       50.12])
latency_jit =      1 / np.array([1.134,                 2.974,             2.796,               2.896,    38.34,  9.738,     2.424,       38.40])
latency_nvfuser =  1 / np.array([1.274,                 8.130,             2.759,               3.523,    48.36,  5.739,     2.691,       43.36])
latency_inductor = 1 / np.array([1.462,                 8.071,             3.134,               2.615,    100.4,  5.299,     2.521,       52.02])
latency_functs =   1 / np.array([0.480,                 1.260,             0.865,               2.123,    19.54,  4.634,     2.048,       11.48])


latency_eager_normal = 1 / latency_eager / latency_eager * latency_eager
latency_jit_normal = 1 / latency_jit / latency_eager * latency_eager
latency_nvfuser_normal = 1 / latency_nvfuser / latency_eager * latency_eager
latency_inductor_normal = 1 / latency_inductor / latency_eager * latency_eager
latency_functs_normal = 1 / latency_functs / latency_eager * latency_eager

data = np.stack([latency_eager_normal, latency_inductor_normal, latency_nvfuser_normal, latency_jit_normal, latency_functs_normal])

# average outperform
outperf_nnc = data[-1] / data[-2]
outperf_total = data[-1] / np.max(data[:-1], axis=0)
print("output perform total: ", outperf_total)
print("output perform total: ", np.mean(outperf_total))
print("output perform nnc: ", outperf_nnc)
print("output perform nnc: ", np.mean(outperf_nnc))

patterns = [ "" , "-" , "\\" , "/" , ".."]
# illustrate bar figure
colors = ["#cbcbcbff", "#e695bfff", "#82c8e1ff", "#8ee4a1ff", "#ffa13ab4",]

bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

fig, axes = plt.subplots(1, latency_eager_normal.shape[0], figsize=(24, 3), layout="constrained")
for b, (c, h) in enumerate(zip(colors, data)):
    for i, hm in enumerate(h): 
        ax = axes[i]
        ax.bar(
            begin_pos + b * bar_width + np.arange(1),
            hm,
            width=bar_width,
            color=c,
            edgecolor="black",
            label="Normalized Performance",
            hatch=patterns[b]
        )
        # axes[i].xticks(range(len(tools)), models, rotation=10)
        ax.get_xaxis().set_visible(False)
        ax.set_title(models[i], fontsize=18, weight="bold", y = -0.15)
        # fig.ylabel("Normalized Performance")
for i, _ in enumerate(models):
    m = data.shape[0] - 1 
    outperf = data[-1, i] / np.min(data[:-1, i])
    axes[i].text(
        0 - begin_pos,
        data[-1, i],
        "{:.2f}x".format(1 / outperf),
        ha="center",
        va="bottom",
        weight='bold',
        fontsize=16
    )
    fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center", fontsize=16, borderpad=0.6)

plt.savefig('latency_3090.pdf')
plt.savefig('latency_3090.jpg')
plt.show()





