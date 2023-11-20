import numpy as np
import matplotlib.pyplot as plt

tools = ["TorchDynamo + Inductor", "TorchScript + nvfuser", "TorchScript + NNC", "TorchScript + TensorSSA + NNC"]
models = ["NASRNN", "Attention", "LSTM", "seq2seq"]

# LSTM
iters = [50, 100, 150, 200, 250,  300]
latency_eager = np.array( [ 98.53,  198.8, 297.1, 399.7, 484.8, 596.5])
latency_jit = np.array(   [ 38.32,  74.67, 112.5, 150.4, 186.6, 224.1])
latency_dynamo = np.array([ 31.34,  61.29, 92.11, 124.0, 155.4, 183.8])
latency_functs = np.array([ 18.88,  37.97, 56.85, 75.4,  93.51, 114.0])
latency_nvfuser = np.array([49.01,  98.33, 146.8, 196.6, 245.1, 294.9])
# latency_tracing_jit = [1.101, 2.858]

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

lstm_data = np.stack([latency_dynamo_normal, latency_nvfuser_normal, latency_jit_normal, latency_functs_normal])

# NASRNN
iters = [50, 100, 150, 200, 250, 300]
latency_eager = np.array(   [12.92, 26.30, 39.46, 52.81, 68.87, 79.24])
latency_jit = np.array(     [2.847, 5.729, 8.493, 11.25, 13.96, 16.82])
latency_dynamo = np.array(  [2.554, 5.137, 7.441, 9.832, 12.07, 17.64])
latency_functs = np.array(  [2.112, 4.243, 6.178, 8.212, 10.12, 12.31])
latency_nvfuser = np.array( [3.495, 6.995, 10.30, 13.75, 17.17, 20.54])
# latency_tracing_jit = [1.101, 2.858]

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

nasrnn_data = np.stack([latency_dynamo_normal, latency_nvfuser_normal, latency_jit_normal, latency_functs_normal])

# # Attention
iters = [50, 100, 150, 200, 250, 300]
latency_eager = np.array(  [3.185, 12.00, 20.82, 29.73, 38.23, 49.69])
latency_jit = np.array(    [2.359, 8.745, 15.1,  22.47, 29.06, 36.05])
latency_dynamo = np.array( [2.448, 9.011, 15.74, 23.19, 29.91, 36.59])
latency_functs = np.array( [1.964, 7.319, 12.69, 18.71, 24.21, 30.05])
latency_nvfuser = np.array([2.626, 9.764, 16.75, 25.0,  32.39, 39.64])
# latency_tracing_jit = [1.101, 2.858]

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

attention_data = np.stack([latency_dynamo_normal, latency_nvfuser_normal, latency_jit_normal, latency_functs_normal])


# # seq2seq
iters = [50, 100, 150, 200, 250, 300]
latency_eager = np.array(  [19.26, 40.30, 58.33, 77.53, 102.9, 115.1])
latency_jit = np.array(    [9.963, 19.49, 28.96, 38.89, 49.09, 58.61])
latency_dynamo = np.array( [5.291, 10.52, 15.04, 20.85, 25.21, 30.18])
latency_functs = np.array( [ 4.56, 9.299, 13.65, 18.37, 22.47, 27.75])
latency_nvfuser = np.array([5.801, 11.58, 16.76, 22.90, 28.41, 34.06])
# latency_tracing_jit = [1.101, 2.858]

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

seq2seq_data = np.stack([latency_dynamo_normal, latency_nvfuser_normal, latency_jit_normal, latency_functs_normal])


data = {
    "NASRNN": nasrnn_data,
    "LSTM": lstm_data,
    "Attention": attention_data,
    "seq2seq": seq2seq_data,
}


# illustrate bar figure
colors = ["#e695bfff", "#82c8e1ff", "#8ee4a1ff", "#ffa13ab4",]
markers = ["s", "D", "*", "8"]
bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
for model, ax in zip(models, axes):
    # fig = plt.figure(figsize=(4, 3), layout='constrained')
    for b, (c, m, h) in enumerate(zip(colors, markers, data[model])):
         ax.plot(
            iters,
            h,
            # width=bar_width,
            color=c,
            marker=m,
            markerfacecolor="white",
            # edgecolor="black",
        )
    ax.grid()
    ax.set_title(model, fontsize=14, weight="bold")
    ax.set_xlabel("Sequence Length", fontsize=13)
    # print(model)
    if model == "Attention":
        ax.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2.0])
        # ax.set_ylim([1.5, np.ceil(np.max(data[model]))])
    elif model == "NASRNN":
        ax.set_yticks([3, 4, 5, 6, 7, 8])
    elif model == "LSTM":
        ax.set_yticks([1, 2, 3, 4, 5, 6])
    elif model == "Seq2seq":
        ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_xticks(iters)
    


# plt.xticks(range(len(iters)), iters, rotation=10)
fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center")
fig.supylabel("Speed Up", weight='bold')

plt.savefig('scalability_seq_len.pdf')
plt.savefig('scalability_seq_len.jpg')
plt.show()
