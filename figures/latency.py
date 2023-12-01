import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", 
         "TorchDynamo + Inductor", 
         "TorchScript + nvfuser", 
         "TorchScript + NNC", 
         "TorchScript + TensorSSA + NNC"]

# 1660Ti
models =                         ["YOLOV3",    "SSD",            "YOLACT",      "FCOS",         "NASRNN", "LSTM", "seq2seq", "Attention"]
latency_eager = 1 / np.array(    [1.199+1.54,  2.869 + 1.35,     4.480 + 3.41,  2.761 + 5.136,  12.68,    58.46,  3.165,     3.79,      ])
latency_jit = 1 / np.array(      [0.837+1.54,  2.079 + 1.35,     3.180 + 3.41,  1.662 + 5.136,  5.45,     26.73,  1.79,      2.72,      ])
latency_nvfuser = 1 / np.array(  [0.964+1.54,  5.905 + 1.35,     3.202 + 3.41,  1.632 + 5.136,  2.44,     31.45,  1.172,     2.733,     ])
latency_inductor = 1 / np.array( [1.106+1.54,  2.847 + 1.35,     2.655 + 3.41,  2.869 + 5.136,  2.44,     23.89,  3.023,     2.65,      ])
# latency_tracing_jit = [1.101, 2.858]
latency_functs = 1 / np.array(   [0.447+1.54,  0.960 + 1.35,     1.110 + 3.41,  1.221 + 5.136,  1.86,     13.38,  0.9857,    2.38,      ])

latency_eager_normal = latency_eager / latency_eager
latency_jit_normal = latency_jit / latency_eager
latency_nvfuser_normal = latency_nvfuser / latency_eager
latency_inductor_normal = latency_inductor / latency_eager
latency_functs_normal = latency_functs / latency_eager

data_1660ti = np.stack([latency_eager_normal,
                        latency_inductor_normal,
                        latency_nvfuser_normal,
                        latency_jit_normal,
                        latency_functs_normal])

# 3090
models =                        ["YOLOV3",     "SSD",        "YOLACT",      "FCOS",                 "NASRNN", "LSTM", "seq2seq", "Attention"]
latency_eager =    1 / np.array([1.443 + 1.49, 4.092 + 1.52, 4.354 + 1.83,  2.77 + 8.75,            13.23,    105.2,  19.13,     3.241,     ])
latency_jit =      1 / np.array([1.134 + 1.49, 2.974 + 1.52, 2.796 + 1.83,  2.55 + 8.75,            2.896,    39.29,  9.738,     2.424,     ])
latency_nvfuser =  1 / np.array([1.274 + 1.49, 8.130 + 1.52, 2.759 + 1.83,  2.55 + 8.75,            3.523,    52.1,   5.739,     2.691,     ])
latency_inductor = 1 / np.array([1.462 + 1.49, 8.071 + 1.52, 3.134 + 1.83,  2.78 + 8.75,            2.615,    31.61,  5.299,     2.521,     ])
latency_functs =   1 / np.array([0.480 + 1.49, 1.260 + 1.52, 0.865 + 1.83,  1.33 + 8.75,            2.123,    19.54,  4.634,     2.048,     ])


latency_eager_normal = latency_eager / latency_eager
latency_jit_normal = latency_jit / latency_eager
latency_nvfuser_normal = latency_nvfuser / latency_eager
latency_inductor_normal = latency_inductor / latency_eager
latency_functs_normal = latency_functs / latency_eager

data_3090 = np.stack([latency_eager_normal, 
                      latency_inductor_normal, 
                      latency_nvfuser_normal, 
                      latency_jit_normal, 
                      latency_functs_normal])

platforms = ["(a) RTX 1660Ti", "(b) RTX 3090"]

data = {
    "(b) RTX 3090": data_3090,
    "(a) RTX 1660Ti": data_1660ti,
}

# average outperform
# outperf_nnc = data[-1] / data[-2]
# outperf_total = data[-1] / np.max(data[:-1], axis=0)
# print("output perform total: ", outperf_total)
# print("output perform total: ", np.mean(outperf_total))
# print("output perform nnc: ", outperf_nnc)
# print("output perform nnc: ", np.mean(outperf_nnc))

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





