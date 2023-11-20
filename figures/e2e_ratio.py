from matplotlib import pyplot as plt
import numpy as np

platforms = ["AGX Xavier", "GTX 1660 Ti", "RTX 3090", "A100"]
models = ["YOLOv3", "SSD", "FCOS", "YOLACT", "SOLOv2"]
tools = ["PyTorch", "nvFuser", "TensorRT", "TorchInductor", "DISC", "Ours"]
colors = [
    "orchid",
    "cornflowerblue",
    "turquoise",
    "mediumseagreen",
    "peachpuff",
    "salmon",
]
post_time = np.array(
    [
        [  # AGX Xavier
            # YOLOv3  SSD  FCOS  YOLACT  SOLOv2
            [6.02, 12.58, 10.36, 22.75, 11.53],  # PyTorch
            [5.27, 37.09, 9.28, 16.03, 13.33],  # nvFuser
            [float("inf"), 18.63, 10.46, 17.91, 13.81],  # TensorRT
            [5.49, 13.89, 13.01, 16.8, 12.24],  # TorchInductor
            [float("inf")] * 5,  # DISC
            [2.47, 5.04, 4.99, 5.92, 9.6],  ## Ours
        ],
        [  # GTX 1660 Ti
            [1.07, 2.85, 1.74, 4.48, 2.27],
            [0.842, 5.99, 1.84, 3.18, 2.85],
            [float("inf"), 6.69, 2.37, 4.02, float("inf")],
            [1.13, 3.16, 1.84, 2.7, 2.29],
            [1, 2.7, 1.93, 4.49, 2.37],
            [0.392, 0.901, 0.987, 1.11, 1.79],
        ],
        [  # RTX 3090
            [1.44, 3.94, 2.77, 4.06, 1.9],
            [1.17, 8.42, 2.55, 2.62, 3.25],
            [float("inf"), 12.26, 6.58, 7.9, 4.18],
            [1.34, 3.7, 2.78, 3.17, 2.11],
            [1.35, 3.83, 2.57, 4.02, 2.13],
            [0.48, 1.26, 1.33, 0.865, 1.17],
        ],
        [
           # A100
            [1.53, 4.02, 2.81, 4.43, 2],
            [1.2, 3.11, 2.57, 2.91,	1.61],
	        [float("inf"), 8.72, 6.42, 7.34, 2.45],
            [1.47, 4.05, 3.58, 3.32, 2.4],
			[float("inf")] * 5,
            [0.53, 1.36, 1.46, 1.24, 1.2],
        ]
    ]
)
assert post_time.shape == (len(platforms), len(tools), len(models))
net_time = np.array(
    [
        # YOLOv3  SSD  FCOS  YOLACT  SOLOv2
        [5.75, 3.9, 47.19, 10.81, 47.9],  # AGX Xavier
        [1.54, 1.35, 10.3, 3.41, 10.5],  # GTX 1660 Ti
        [1.49, 1.52, 6.44, 1.83, 4.9],  # RTX 3090
        [1.32, 1.33, 6.64, 1.65, 4.47],  # A100
    ]
)
assert net_time.shape == (len(platforms), len(models))

bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

# Post-processing
fig, axes = plt.subplots(1, 4, figsize=(16, 3), layout="constrained", sharey=True)
max_ylim = -1
for plat, data, ax in zip(platforms, post_time, axes):
    perf = data[0:1] / data
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
fig.legend(labels=tools, ncol=len(tools), loc="outside upper center")
fig.supylabel("Normalized Performance", weight='bold')

plt.savefig("e2e_post.pdf")

# Whole model
fig, axes = plt.subplots(1, 4, figsize=(16, 3), layout="constrained", sharey=True)
for plat, data, net, ax in zip(platforms, post_time, net_time, axes):
    total_time = data + net[None, :]
    perf = total_time[0:1] / total_time
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
    ax.set_ylim(top=np.ceil(np.max(perf * 2)) / 2)
fig.legend(labels=tools, ncol=len(tools), loc="outside upper center")
fig.supylabel("Normalized Performance")

plt.savefig("e2e_model.pdf")
plt.show()