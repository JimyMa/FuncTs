from matplotlib import pyplot as plt
import numpy as np


platforms = ['GTX 1660 Ti', 'RTX 3090',]
colors = ['lightskyblue', 'lightsalmon']
models = ['YOLOv3', 'SSD', 'YOLACT', "FCOS", "NASRNN",    "LSTM",    "seq2seq", "Attention"]
data = np.array([
        [0.35,      0.67,  0.44,     0.14,   293. / 363., 185 / 261., 410 / 570., 130 / 270.9],
        [0.44,      0.71,  0.59,     0.28,   453. / 504., 310 / 393., 480 / 649., 170 / 331.],
])
assert len(platforms) == len(colors) == data.shape[0]
assert len(models) == data.shape[1]

bar_width = 1 / (len(platforms) + 2)

plt.rc('font', family='Linux Biolinum O', size=12)
plt.rc('pdf', fonttype=42)

plt.figure(figsize=(7, 3), constrained_layout=True)
begin_pos = -(len(platforms) - 1) * bar_width / 2
for i, (c, h) in enumerate(zip(colors, data)):
    plt.bar(begin_pos + i * bar_width + np.arange(len(models)),
            h, width=bar_width * .8, color=c, 
        #     edgecolor='black'
            )
plt.legend(labels=platforms, ncol=1, loc='best')
plt.xticks(range(len(models)), models)
plt.ylim(top=1)

plt.savefig('post_ratio.pdf')
plt.show()

