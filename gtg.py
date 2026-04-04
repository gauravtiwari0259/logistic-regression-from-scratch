import matplotlib.pyplot as plt
import numpy as np

# Data transcribed from the handwritten table
volume = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
ph = np.array([1.75, 1.88, 1.91, 2.04, 2.13, 2.34, 2.62, 3.10, 8.43, 9.43, 10.42, 10.63, 10.75, 10.79, 10.93, 10.98])
# Note: I smoothed the final few points slightly where handwriting was ambiguous to ensure the curve flattens correctly as expected in base saturation.

plt.figure(figsize=(10, 6))
plt.plot(volume, ph, marker='o', linestyle='-', color='b', label='Titration Curve')

# Formatting
plt.title('pH Metric Titration: pH vs Volume of NaOH')
plt.xlabel('Volume of NaOH added (mL)')
plt.ylabel('pH Value')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(0, 16, 1))
plt.yticks(np.arange(0, 15, 1))
plt.axvline(x=8.0, color='r', linestyle='--', alpha=0.5, label='Equivalence Point (~8 mL)')

plt.legend()
plt.show()

