from matplotlib import pyplot as plt
import numpy as np


platforms = ['Vertical', 'Vertical + Horizontal',]
colors = ['lightskyblue', "#ffa13ab4"]
models = ['YOLOv3', 'SSD', 'YOLACT', "FCOS"]



data = np.array([
    [(1.443 + 1.49) / (0.549 + 1.49), (4.092 + 1.52) / (1.491 + 1.52), (4.354 + 1.83) / (1.630 + 1.83),  (2.22 + 8.75) / (1.058 + 8.75)],
    [(1.443 + 1.49) / (0.480 + 1.49), (4.092 + 1.52) / (1.260 + 1.52), (4.354 + 1.83) / (0.865 + 1.83),  (2.22 + 8.75) / (0.867 + 8.75)],
])
assert len(platforms) == len(colors) == data.shape[0]
assert len(models) == data.shape[1]

bar_width = 1 / (len(platforms) + 2)

plt.rc('font', family='Linux Biolinum O', size=12)
plt.rc('pdf', fonttype=42)

plt.figure(figsize=(9, 3), constrained_layout=True)
begin_pos = -(len(platforms) - 1) * bar_width / 2
for i, (c, h) in enumerate(zip(colors, data)):
    plt.bar(begin_pos + i * bar_width + np.arange(len(models)),
            h, width=bar_width * .8, color=c, 
        #     edgecolor='black'
            )
plt.legend(labels=platforms, ncol=1, loc='best')
plt.xticks(range(len(models)), models)
# plt.ylim(top=1)

plt.savefig('ablation.pdf')
plt.savefig('ablation.jpg')
plt.show()

