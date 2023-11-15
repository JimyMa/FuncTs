import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", "TorchDynamo", "nvfuser", "TorchScript", "FuncTs"]
models = ["NASRNN", "Attention", "LSTM"]

# LSTM
iters = [1.0, 2.0, 3.0, 4.0, 5.0]
latency_eager = np.array([82.22, 71.55, 70.12, 66.53, 66.61])
latency_jit = np.array([28.36, 28.27, 28.23, 29.44, 28.801])
latency_dynamo = np.array([0, 0, 0, 0, 0])
latency_functs = np.array([15.56, 15.47, 16.14, 15.71, 15.78])
latency_nvfuser = np.array([37.83, 37.52, 38.73, 38.36, 38.57])
# latency_tracing_jit = [1.101, 2.858]

latency_eager_normal = latency_eager
latency_jit_normal = latency_jit
latency_functs_normal = latency_functs

lstm_data = np.stack([latency_eager_normal, latency_dynamo, latency_nvfuser, latency_jit_normal, latency_functs_normal])

# NASRNN
iters = [1.0, 2.0, 3.0, 4.0, 5.0]
latency_eager = np.array([7.21, 7.14, 7.02, 7.06, 7.023])
latency_jit = np.array([1.67, 1.67, 1.66, 1.67, 1.66])
latency_dynamo = np.array([1.151, 1.118, 1.13, 1.137, 1.129])
latency_functs = np.array([0.854, 0.838, 0.834, 0.863, 0.838])
latency_nvfuser = np.array([2.332, 2.274, 2.25, 2.275, 2.248])
# latency_tracing_jit = [1.101, 2.858]

latency_eager_normal = latency_eager
latency_jit_normal = latency_jit
latency_functs_normal = latency_functs

nasrnn_data = np.stack([latency_eager_normal, latency_dynamo, latency_nvfuser, latency_jit_normal, latency_functs_normal])

# Attention
iters = [1.0, 2.0, 3.0, 4.0, 5.0]
latency_eager = np.array([3.317, 3.328, 3.394, 3.463, 3.44])
latency_dynamo = np.array([2.306, 2.309, 2.3, 2.367, 2.356])
latency_jit = np.array([2.397, 2.363, 2.395, 2.445, 2.484])
latency_functs = np.array([2.014, 2.013, 2.005, 2.008, 2.008])
latency_nvfuser = np.array([2.384, 2.37, 2.4, 2.395, 2.38])
# latency_tracing_jit = [1.101, 2.858]

latency_eager_normal = latency_eager
latency_jit_normal = latency_jit

latency_functs_normal = latency_functs


attention_data = np.stack([latency_eager_normal, latency_dynamo, latency_nvfuser, latency_jit_normal, latency_functs_normal])

data = {
    "NASRNN": nasrnn_data,
    "LSTM": lstm_data,
    "Attention": attention_data,
}


# illustrate bar figure
colors = ["#003366", "#E31B23", "#005CAB", "#784c22", "#FFC325",]
markers = ["o", "s", "D", "*", "8"]
bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), layout="constrained")
for model, ax in zip(models, axes):
    # fig = plt.figure(figsize=(4, 3), layout='constrained')
    for b, (c, m, h) in enumerate(zip(colors, markers, data[model])):
        if not (model == "LSTM" and b == 1):
            ax.plot(
                iters if model != "Attention" else [iter + 10 for iter in iters],
                h,
                # width=bar_width,
                color=c,
                marker=m,
                markerfacecolor="white",
                # edgecolor="black",
            )
    ax.grid()
    ax.set_title(model, fontsize=14, weight="bold")
    ax.set_xlabel("Iters / Capture")
    # print(model)
    if model == "Attention":
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        # ax.set_ylim([1.5, np.ceil(np.max(data[model]))])
    elif model == "NASRNN":
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_xlim(0.88, 5.12)
    else:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(0.88, 5.12)
    


# plt.xticks(range(len(iters)), iters, rotation=10)
fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center")
fig.supylabel("Latency (ms)", weight='bold')

plt.savefig('latency_cudagraph.pdf')
plt.savefig('latency_cudagraph.jpg')
plt.show()