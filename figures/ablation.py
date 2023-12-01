from matplotlib import pyplot as plt
import numpy as np


platforms = ["PyTorchEager", "TorchScript + NNC", 'TorchScript + TensorSSA + NNC (Vertical)', 'TorchScript + TensorSSA + NNC (Vertical + Horizontal)',]
colors = ["#cbcbcbff", "#8ee4a1ff", 'lightskyblue', "#ffa13ab4"]
models = ['YOLOv3', 'SSD', 'YOLACT', "FCOS"]



data = np.array([
    [1, 1, 1,  1],
    [(1.443 + 1.49) / (1.080 + 1.49), (4.092 + 1.52) / (2.710 + 1.52), (4.354 + 1.83) / (2.932 + 1.83),  (2.22 + 8.75) / (1.589 + 8.75)],
    [(1.443 + 1.49) / (0.549 + 1.49), (4.092 + 1.52) / (1.491 + 1.52), (4.354 + 1.83) / (1.630 + 1.83),  (2.22 + 8.75) / (1.058 + 8.75)],
    [(1.443 + 1.49) / (0.480 + 1.49), (4.092 + 1.52) / (1.260 + 1.52), (4.354 + 1.83) / (0.865 + 1.83),  (2.22 + 8.75) / (0.867 + 8.75)],
])
assert len(platforms) == len(colors) == data.shape[0]
assert len(models) == data.shape[1]

bar_width = 1 / (len(platforms) + 2)

plt.rc('font', family='Linux Biolinum O', size=12)
plt.rc('pdf', fonttype=42)

plt.figure(figsize=(14, 4), constrained_layout=True)
begin_pos = -(len(platforms) - 1) * bar_width / 2
for i, (c, h) in enumerate(zip(colors, data)):
    plt.bar(begin_pos + i * bar_width + np.arange(len(models)),
            h, width=bar_width * .8, color=c, 
        #     edgecolor='black'
            )
plt.legend(labels=platforms, ncol=1, loc='best')
plt.axhline(y = 1.0, 
            color = "#cbcbcbff", 
            linestyle = ':') 
plt.xticks(range(len(models)), models)
plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
# plt.ylim(top=1)

plt.savefig('ablation.pdf')
plt.savefig('ablation.jpg')
plt.show()

