import matplotlib.pyplot as plt
import numpy as np
d_ph = np.diff(ph)
d_v = np.diff(volume)
derivative = d_ph / d_v
mid_points = volume[1:] # Plotting at the volume where the change was measured

plt.figure(figsize=(10, 6))
plt.plot(mid_points, derivative, marker='x', linestyle='-', color='green', label='First Derivative')

# Formatting
plt.title('First Derivative: $\Delta$pH / $\Delta$V')
plt.xlabel('Volume of NaOH added (mL)')
plt.ylabel('$\Delta$pH / $\Delta$V')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(0, 16, 1))

# Highlight the peak
peak_index = np.argmax(derivative)
peak_vol = mid_points[peak_index]
plt.annotate(f'Peak at {peak_vol} mL', xy=(peak_vol, derivative[peak_index]), xytext=(peak_vol+1, derivative[peak_index]),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()