import numpy as np
import matplotlib.pyplot as plt

tools = ["PyTorch Eager", 
         "TorchDynamo + Inductor",  
         "TorchScript + nvfuser", 
         "TorchScript + NNC", 
         "TorchScript + TensorSSA + NNC"]

models =                  ["YOLOV3", "SSD", "YOLACT", "FCOS", "NASRNN", "LSTM", "seq2seq", "Attention"]
latency_eager = np.array(  [96,      242,   269 ,     268,    1655,     10822,  1449,      323,         ])
latency_jit = np.array(    [83,      236,   231,      185,    154,      3202,   401,       322,         ])
latency_dynamo = np.array( [64,      161,   147,      124,    106,      3123,   301,       288,         ])
latency_functs = np.array( [34,      103,   92,       73,     104,      1921,   351,       288,         ])
latency_nvfuser = np.array([75,      187,   181,      182,    203,      3200,   352,       321,         ])


# latency_tracing_jit = [1.101, 2.858]


latency_eager_normal = latency_eager / latency_eager * latency_eager
latency_jit_normal = latency_jit / latency_eager * latency_eager
latency_functs_normal = latency_functs / latency_eager * latency_eager

latency_nvfuser_normal = latency_nvfuser / latency_eager * latency_eager
latency_dynamo_normal = latency_dynamo / latency_eager * latency_eager

data = np.stack([latency_eager_normal, 
                 latency_dynamo_normal, 
                 latency_nvfuser_normal, 
                 latency_jit_normal, 
                 latency_functs_normal])

# illustrate bar figure
colors = ["orchid", "cornflowerblue", "salmon",]
# colors = ["#003366", "#E31B23", "#005CAB", "#784c22", "#FFC325",]
colors = ["#cbcbcbff", 
          "#e695bfff", 
          "#82c8e1ff", 
          "#8ee4a1ff", 
          "#ffa13ab4",]
patterns = [ "" ,
             "-" , 
             "\\" , 
             "/" , 
             ".."]

bar_width = 1 / (len(tools) + 2)
begin_pos = -(len(tools) - 1) * bar_width / 2

plt.rc("font", family="Linux Biolinum O", size=16)
plt.rc("pdf", fonttype=42)

base = 4

fig, axes = plt.subplots(int(latency_eager_normal.shape[0] / base), base, figsize=(12, 6), layout="constrained")
# ax_big = fig.add_subplot(111)  
# fig = plt.figure(figsize=(14, 3), layout='constrained')
for b, (c, h) in enumerate(zip(colors, data)):
    for i, hm in enumerate(h):
        print(axes.shape)
        print(i)
        print(i % base, i // base)
        ax = axes[ i // base, i % base]
       
        ax.bar(
            begin_pos + b * bar_width + np.arange(1),
            hm,
            width=bar_width * 0.8,
            color=c,
            edgecolor="black",
            label="Normalized Performance",
            hatch=patterns[b],
        )
        ax.get_xaxis().set_visible(False)
        ax.set_title(models[i], fontsize=17, weight="bold", y = -0.15)

# ax = fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
fig.supylabel("Kernel Launches", weight='bold')
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)


fig.legend(labels=tools, ncol=int(np.ceil(len(tools) / 2)), loc="outside upper center")

plt.savefig('kernel_launch.pdf')
plt.savefig('kernel_launch.jpg')
plt.show()