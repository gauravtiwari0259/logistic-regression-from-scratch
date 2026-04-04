import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel("X values (radians)")
plt.ylabel("Y = sin(x)")
plt.title("Sine Wave")
plt.grid(True)

plt.show()   # 👈 This line actually displays the graph in PyCharm

