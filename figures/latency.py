import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", "TorchDynamo + Inductor", "TorchScript + nvfuser", "TorchScript + NNC", "FuncTs"]

models = ["YOLOV3_postprocess", "YOLOV3", "SSD_postprocess", "SSD", "YOLACT_postprocess", "YOLACT", "NASRNN", "LSTM", "seq2seq", "Attention", "Normalize"]
latency_eager = 1 / np.array([1.199, 1.199+1.54, 2.869, 2.869 + 1.35, 4.48, 4.48 + 3.41, 12.68, 92.58, 3.165, 3.79, 0.168])
latency_jit = 1 / np.array([0.837, 0.837+1.54, 2.079, 2.079 + 1.35, 3.18, 3.18 + 3.41, 5.45, 36.59, 1.79, 2.72, 0.141])
latency_nvfuser = 1 / np.array([0.964, 0.964+1.54, 5.905, 5.905 + 1.35, 3.207, 3.202 + 3.41, 2.44, 48.35, 1.172, 2.733, 0.1433])
latency_inductor = 1 / np.array([1.106, 1.106+1.54, 2.847, 2.847 + 1.35, 2.655, 2.655 + 3.41, 2.44, 91.33, 3.023, 2.65, 0.04727])
# latency_tracing_jit = [1.101, 2.858]
latency_functs = 1 / np.array([0.447, 0.447+1.54, 0.960, 0.960 + 1.35, 1.11, 1.11+3.41, 1.86, 18.48, 0.9857, 2.38, 0.03733])


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
        weight='bold'
    )
    fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center", fontsize=16, borderpad=0.6)

plt.savefig('latency.pdf')
plt.savefig('latency.jpg')
plt.show()





