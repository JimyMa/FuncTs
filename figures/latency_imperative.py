import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", "TorchDynamo + Inductor", "TorchScript + nvfuser", "TorchScript + NNC", "FuncTs"]

# 1660Ti
models =                         ["YOLOV3",            "SSD",             "YOLACT",          "NASRNN", "LSTM", "seq2seq", "Attention", "Normalize"]
latency_eager = 1 / np.array(    [1.199,               2.869,             4.48,              12.68,    92.58,  3.165,     3.79,        0.167])
latency_jit = 1 / np.array(      [0.837,               2.079,             3.18,              5.45,     36.59,  1.79,      2.72,        0.141])
latency_nvfuser = 1 / np.array(  [0.964,               5.905,             3.202,             2.44,     48.35,  1.172,     2.733,       0.149])
latency_inductor = 1 / np.array( [1.106,               2.847,             2.655,             2.44,     91.33,  3.023,     2.65,        0.050])
# latency_tracing_jit = [1.101, 2.858]
latency_functs = 1 / np.array(   [0.447,               0.960,             1.11,              1.86,     18.48,  0.9857,    2.38,        0.037])

latency_eager_normal = latency_eager / latency_eager
latency_jit_normal = latency_jit / latency_eager
latency_nvfuser_normal = latency_nvfuser / latency_eager
latency_inductor_normal = latency_inductor / latency_eager
latency_functs_normal = latency_functs / latency_eager

data_1660ti = np.stack([latency_eager_normal, latency_inductor_normal, latency_nvfuser_normal, latency_jit_normal, latency_functs_normal])

# 3090
models =                        ["YOLOV3",              "SSD",             "YOLACT",            "NASRNN", "LSTM", "seq2seq", "Attention", "Normalize"]
latency_eager =    1 / np.array([1.443,                 4.092,             4.354,               0.314,    99.93,  19.13,     3.241,       50.12])
latency_jit =      1 / np.array([1.134,                 2.974,             2.796,               0.56,    38.34,  9.738,     2.424,       38.40])
latency_nvfuser =  1 / np.array([1.274,                 8.130,             2.759,               0.29,    48.36,  5.739,     2.691,       43.36])
latency_inductor = 1 / np.array([1.462,                 8.071,             3.134,               2.615,    100.4,  5.299,     2.521,       52.02])
latency_functs =   1 / np.array([0.480,                 1.260,             0.865,               2.123,    19.54,  4.634,     2.048,       11.48])


latency_eager_normal = latency_eager / latency_eager
latency_jit_normal = latency_jit / latency_eager
latency_nvfuser_normal = latency_nvfuser / latency_eager
latency_inductor_normal = latency_inductor / latency_eager
latency_functs_normal = latency_functs / latency_eager

data_3090 = np.stack([latency_eager_normal, latency_inductor_normal, latency_nvfuser_normal, latency_jit_normal, latency_functs_normal])

platforms = ["RTX 1660Ti", "RTX 3090"]

data = {
    "RTX 3090": data_3090,
    "RTX 1660Ti": data_1660ti,
}


# average outperform
# outperf_nnc = data[-1] / data[-2]
# outperf_total = data[-1] / np.max(data[:-1], axis=0)
# print("output perform total: ", outperf_total)
# print("output perform total: ", np.mean(outperf_total))
# print("output perform nnc: ", outperf_nnc)
# print("output perform nnc: ", np.mean(outperf_nnc))

patterns = [ "" , "-" , "o" , "." , "*"]
# illustrate bar figure
colors = ["#cbcbcbff", "#e695bfff", "#82c8e1ff", "#8ee4a1ff", "#ffa13ab4",]

bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

# Post-processing
fig, axes = plt.subplots(1, 2, figsize=(18, 3), layout="constrained")
max_ylim = -1
for plat, ax in zip(platforms, axes):
    perf = data[plat]
    print(perf.shape)
    print(len(colors))
    for b, (c, h) in enumerate(zip(colors, perf)):
        ax.bar(
            begin_pos + b * bar_width + np.arange(len(models)),
            h,
            width=bar_width,
            color=c,
            edgecolor="black",
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
    ax.set_xticks(range(len(models)), models)
    ax.set_xlabel(plat, fontsize=14, weight="bold")
    max_ylim = max(max_ylim, float(np.ceil(np.max(perf))))
    plt.setp(axes, ylim=(0, max_ylim))
# fig.legend(labels=tools, ncol=len(tools), loc="outside upper center")
fig.supylabel("Normalized Performance", weight='bold')

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
fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center", fontsize=16)
fig.supylabel("Normalized Performance")
plt.savefig('latency_imperative.pdf')
plt.savefig('latency_imperative.jpg')
plt.show()



