import numpy as np
import matplotlib.pyplot as plt

tools = [
    "TorchDynamo + Inductor",
    "TorchScript + nvfuser",
    "TorchScript + NNC",
    "TorchScript + TensorSSA + NNC"
]
models = ["YOLOV3", "SSD", "YOLACT", "FCOS", "NASRNN", "Attention", "LSTM", "seq2seq"]

# FCOS
iters = [1, 2, 4, 6, 8]
latency_eager = np.array(    [2.761 + 5.136, 3.508 + 7.406, 6.905 + 11.97, 10.64 + 16.85, 13.66 + 20.8])
latency_jit = np.array(      [1.662 + 5.136, 2.932 + 7.406, 5.890 + 11.97, 9.260 + 16.85, 11.64 + 20.8])
latency_dynamo = np.array(   [2.869 + 5.136, 5.818 + 7.406, 11.83 + 11.97, 16.72 + 16.85, 22.60 + 20.8])
latency_functs = np.array(   [1.221 + 5.136, 1.374 + 7.406, 1.538 + 11.97, 1.634 + 16.85, 1.790 + 20.8])
latency_nvfuser = np.array(  [1.632 + 5.136, 3.093 + 7.406, 5.856 + 11.97, 10.88 + 16.85, 11.39 + 20.8])

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

fcos_data = np.stack([
    latency_dynamo_normal, 
    latency_nvfuser_normal, 
    latency_jit_normal, 
    latency_functs_normal])

# SSD
iters = [1, 2, 4, 6, 8]
latency_eager = np.array(    [2.892 + 1.35, 5.287 + 1.962, 10.69+ 3.053, 15.61 + 4.031, 19.67 + 5.453])
latency_jit = np.array(      [2.131 + 1.35, 3.822 + 1.962, 7.451+ 3.053, 11.15 + 4.031, 14.02 + 5.453])
latency_dynamo = np.array(   [4.619 + 1.35, 7.449 + 1.962, 10.58+ 3.053, 25.04 + 4.031, 33.52 + 5.453])
latency_functs = np.array(   [0.977 + 1.35, 1.827 + 1.962, 3.475+ 3.053, 5.502 + 4.031, 7.002 + 5.453])
latency_nvfuser = np.array(  [6.138 + 1.35, 9.237 + 1.962, 16.82+ 3.053, 14.22 + 4.031, 16.37 + 5.453])

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

ssd_data = np.stack([
    latency_dynamo_normal, 
    latency_nvfuser_normal, 
    latency_jit_normal, 
    latency_functs_normal])

# YOLOV3
iters = [1, 2, 4, 6, 8]
latency_eager = np.array(  [1.199+1.54, 1.666 + 2.405, 2.804 + 4.042, 3.994 + 5.699, 5.143 + 7.506])
latency_jit = np.array(    [0.818+1.54, 1.295 + 2.405, 2.337 + 4.042, 3.361 + 5.699, 4.364 + 7.506])
latency_dynamo = np.array( [1.077+1.54, 2.108 + 2.405, 3.622 + 4.042, 5.244 + 5.699, 6.616 + 7.506])
latency_functs = np.array( [0.489+1.54, 0.860 + 2.405, 1.624 + 4.042, 2.369 + 5.699, 3.146 + 7.506])
latency_nvfuser = np.array([1.077+1.54, 1.425 + 2.405, 2.524 + 4.042, 3.583 + 5.699, 4.678 + 7.506])
# latency_tracing_jit = [1.101, 2.858]

latency_jit_normal = latency_eager / latency_jit
latency_dynamo_normal = latency_eager / latency_dynamo
latency_functs_normal = latency_eager / latency_functs
latency_nvfuser_normal = latency_eager / latency_nvfuser

yolov3_data = np.stack([
    latency_dynamo_normal, 
    latency_nvfuser_normal, 
    latency_jit_normal, 
    latency_functs_normal])

# # YOLACT
iters = [1, 2, 4, 6, 8]
latency_eager = np.array(  [4.353 + 1.83, 8.315 + 5.857, 16.79 + 10.74, 24.29 + 15.02, 28.01 + 18.07])
latency_jit = np.array(    [2.680 + 1.83, 5.060 + 5.857, 10.25 + 10.74, 14.82 + 15.02, 17.26 + 18.07])
latency_dynamo = np.array( [2.697 + 1.83, 4.980 + 5.857, 9.932 + 10.74, 14.86 + 15.02, 16.83 + 18.07])
latency_functs = np.array( [1.240 + 1.83, 2.098 + 5.857, 4.228 + 10.74, 6.327 + 15.02, 7.155 + 18.07])
latency_nvfuser = np.array([3.307 + 1.83, 6.484 + 5.857, 12.30 + 10.74, 18.11 + 15.02, 20.57 + 18.07])
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
    "YOLOV3": yolov3_data,
    "SSD": ssd_data,
    "YOLACT": attention_data,
    "FCOS": fcos_data,
    "NASRNN": nasrnn_data,
    "LSTM": lstm_data,
    "Attention": attention_data,
    "seq2seq": seq2seq_data,
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

base = 4
fig, axes = plt.subplots(int(len(models) // base), base, figsize=(12, 6.0), layout="constrained")
for i, model in enumerate(models):
    ax = axes[ i // base, i % base]
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
    if model == "YOLOV3":
        ax.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
        # ax.set_ylim([1.5, np.ceil(np.max(data[model]))])
    elif model == "SSD":
        ax.set_yticks([0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
    elif model == "YOLACT":
        ax.set_yticks([1., 1.2, 1.4, 1.6, 1.8, 2.0])
    elif model == "FCOS":
        ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    elif model == "Attention":
        ax.set_yticks([1., 1.2, 1.4, 1.6, 1.8])
        # ax.set_ylim([1.5, np.ceil(np.max(data[model]))])
    elif model == "NASRNN":
        ax.set_yticks([ 3, 4, 5, 6, 7])
    elif model == "LSTM":
        ax.set_yticks([2., 3., 4., 5., 6.0])
    elif model == "Seq2seq":
        ax.set_yticks([1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticks(iters)
    


# plt.xticks(range(len(iters)), iters, rotation=10)
fig.legend(labels=tools, ncol=np.ceil(len(tools)), loc="outside upper center")
fig.supylabel("Speed Up", weight='bold')

plt.savefig('scalability_bs.pdf')
plt.savefig('scalability_bs.jpg')
plt.show()
