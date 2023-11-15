import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

size = 3
x = np.arange(size)

plt.rc("font", family="Linux Biolinum O", size=12)
plt.rc("pdf", fonttype=42)

fig = plt.figure()

tssa = np.array([1.22, 1.25, 1.22])
fait = np.array([1.80, 2.17, 1.80])
colors = ["blue", "#FFC325"]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2


plt.bar(x, tssa,  width=width, label='TorchScript + TensorSSA + NNC')
plt.bar(x + width, fait, width=width, label='FuncTs', color="#FFC325")

# plt.xaxis().set_visible(False)
# plt.set_title(models[i], fontsize=18, weight="bold", y = -0.15)
plt.xticks(x, ["YOLOV3_postprocess", "SSD_postprocess", "YOLACT_postprocess"], rotation=10, weight="bold")

plt.ylabel(ylabel="Normalized Performance", weight="bold")

fig.legend(labels=['TorchScript + TensorSSA + NNC', 'FuncTs'], ncol=np.ceil(2), loc="outside upper center", fontsize=14)

plt.savefig('functs_vs_tensor_ssa.pdf')
plt.show()

