import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch", "TorchScript", "FuncTs"]
models = ["NASRNN", "Attention", "LSTM"]

# LSTM
iters = [1.0, 2.0, 3.0, 4.0, 5.0]
latency_eager = np.array([82.22, 71.55, 70.12, 66.53, 66.61])
latency_jit = np.array([44.07, 32.97, 29.85, 28.87, 27.26])
latency_functs = np.array([33.96, 22.6, 18.83, 16.98, 15.35])
# latency_tracing_jit = [1.101, 2.858]

latency_eager_normal = latency_eager
latency_jit_normal = latency_jit
latency_functs_normal = latency_functs

lstm_data = np.stack([latency_eager_normal, latency_jit_normal, latency_functs_normal])

# NASRNN
iters = [1.0, 2.0, 3.0, 4.0, 5.0]
latency_eager = np.array([87.42, 82.01, 74.98, 72.91, 71.37])
latency_jit = np.array([46.67, 35.72, 32.16, 30.5, 29.1])
latency_functs = np.array([27.9, 16.48, 12.4, 10.36, 9.27])
# latency_tracing_jit = [1.101, 2.858]

latency_eager_normal = latency_eager
latency_jit_normal = latency_jit
latency_functs_normal = latency_functs

nasrnn_data = np.stack([latency_eager_normal, latency_jit_normal, latency_functs_normal])

# Attention
iters = [1.0, 2.0, 3.0, 4.0, 5.0]
latency_eager = np.array([8.25, 8.175, 8.139, 8.122, 8.021])
latency_jit = np.array([6.784, 6.775, 6.732, 6.608, 6.624])
latency_functs = np.array([5.3, 5.216, 5.161, 5.13, 5.116])
# latency_tracing_jit = [1.101, 2.858]

latency_eager_normal = latency_eager
latency_jit_normal = latency_jit
latency_functs_normal = latency_functs

attention_data = np.stack([latency_eager_normal, latency_jit_normal, latency_functs_normal])

data = {
    "NASRNN": nasrnn_data,
    "LSTM": lstm_data,
    "Attention": attention_data,
}


# illustrate bar figure
colors = ["orchid", "cornflowerblue", "salmon",]
markers = ["o", "s", "D"]
bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), layout="constrained")
for model, ax in zip(models, axes):
    # fig = plt.figure(figsize=(4, 3), layout='constrained')
    for b, (c, m, h) in enumerate(zip(colors, markers, data[model])):
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
    if model == "Attention":
        ax.set_xlim(0.88 + 10, 5.12 + 10)
        ax.set_ylim([np.floor(np.min(data[model])), np.ceil(np.max(data[model]))])
    else:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(0.88, 5.12)
    


# plt.xticks(range(len(iters)), iters, rotation=10)
fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center")
fig.supylabel("Latency (ms) / Iter", weight='bold')

plt.savefig('latency_cudagraph.pdf')
plt.savefig('latency_cudagraph.jpg')
plt.show()