import numpy as np
import matplotlib.pyplot as plt

tools = [
    "TorchDynamo + Inductor",
    "TorchScript + nvfuser",
    "TorchScript + NNC",
    "TorchScript + TensorSSA + NNC"
]
models = ["NASRNN", "Attention", "LSTM", "Seq2seq"]

# LSTM
iters = [1, 4, 8, 12, 16]
latency_eager = np.array(    [100.2, 102.8, 99.97, 100.1, 101.4])
latency_jit = np.array(      [37.95, 38.99, 39.45, 40.36, 40.72])
latency_dynamo = np.array(   [30.83, 21.07, 21.19, 21.02, 21.06])
latency_functs = np.array(   [19.06, 19.73, 19.97, 20.28, 19.92])
latency_nvfuser = np.array(  [49.38, 55.16, 51.43, 51.15, 51.21])

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

lstm_data = np.stack([
    latency_dynamo_normal, 
    latency_nvfuser_normal, 
    latency_jit_normal, 
    latency_functs_normal])

# NASRNN
iters = [1, 4, 8, 12, 16]
latency_eager = np.array(  [12.21, 11.78, 12.3,  12.74, 12.46])
latency_jit = np.array(    [2.866, 2.839, 2.872, 2.855, 2.863])
latency_dynamo = np.array( [2.561, 2.568, 2.621, 2.558, 2.565])
latency_functs = np.array( [2.118, 2.115, 2.133, 2.127, 2.137])
latency_nvfuser = np.array([3.510, 3.480, 3.512, 3.506, 3.497])
# latency_tracing_jit = [1.101, 2.858]

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

nasrnn_data = np.stack([
    latency_dynamo_normal, 
    latency_nvfuser_normal, 
    latency_jit_normal, 
    latency_functs_normal])

# # Attention
iters = [1, 4, 8, 12, 16]
latency_eager = np.array(  [3.225, 3.933, 3.948, 3.889, 3.987])
latency_jit = np.array(    [2.371, 3.157, 3.085, 3.096, 3.139])
latency_dynamo = np.array( [2.517, 2.788, 2.781, 2.741, 2.844])
latency_functs = np.array( [1.964, 2.67,  2.68,  2.651, 2.659])
latency_nvfuser = np.array([2.599, 3.31, 3.294 , 3.321, 3.344])
# latency_tracing_jit = [1.101, 2.858]

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

attention_data = np.stack([
    latency_dynamo_normal, 
    latency_nvfuser_normal, 
    latency_jit_normal, 
    latency_functs_normal])


# # seq2seq
iters = [1, 4, 8, 12, 16]
latency_eager = np.array(  [19.42, 22.21, 25.75, 26.21, 25.94])
latency_jit     = np.array([9.924, 10.48, 13.51, 13.27, 13.26])
latency_dynamo  = np.array([5.394, 5.493, 7.361, 7.357, 7.677])
latency_functs  = np.array([4.690, 4.739, 4.963, 4.839, 4.774])
latency_nvfuser = np.array([5.754, 5.862, 6.137, 5.810, 5.824])

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

seq2seq_data = np.stack([
    latency_dynamo_normal, 
    latency_nvfuser_normal, 
    latency_jit_normal, 
    latency_functs_normal])


data = {
    "NASRNN": nasrnn_data,
    "LSTM": lstm_data,
    "Attention": attention_data,
    "Seq2seq": seq2seq_data,
}


# illustrate bar figure
colors = [
    "#e695bfff", 
    "#82c8e1ff", 
    "#8ee4a1ff", 
    "#ffa13ab4",]
markers = [
    "s", 
    "D", 
    "*", 
    "8"]
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
    ax.set_xlabel("Batch Size", fontsize=13)
    # print(model)
    if model == "Attention":
        ax.set_yticks([1., 1.2, 1.4, 1.6, 1.8])
        # ax.set_ylim([1.5, np.ceil(np.max(data[model]))])
    elif model == "NASRNN":
        ax.set_yticks([ 3, 4, 5, 6, 7])
    elif model == "LSTM":
        ax.set_yticks([1., 2., 3., 4., 5., 6.0])
    else:
        ax.set_yticks([1.5, 3.0, 4.5, 6.0])
    ax.set_xticks([0, 4, 8, 12, 16])
    


# plt.xticks(range(len(iters)), iters, rotation=10)
fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center")
fig.supylabel("Speed Up", weight='bold')

plt.savefig('scalability_bs.pdf')
plt.savefig('scalability_bs.jpg')
plt.show()
