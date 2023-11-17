import matplotlib.pyplot as plt
import numpy as np
from Curve import Curve

"""
Here, I am testing the Curve class to make sure the comparisons and fit function work as intended. """

# Generate 20 points for the base fourth degree polynomial curve
x_A4 = np.linspace(-2, 2, 20)
y_A4 = x_A4**4 - 3*x_A4**3 + 2*x_A4**2 + 4*x_A4 + 1

# Curve B4 is identical to Curve A4
x_B4 = x_A4
y_B4 = y_A4

# Curve C4 is Curve A4 shifted up (by 10) and to the right (by 1)
x_C4 = x_A4 + 1
y_C4 = y_A4 + 10

# Curve D4 is Curve A4 inverted (reflected across the x-axis)
x_D4 = x_A4
y_D4 = -y_A4

# Generate random data that has no real underlying function
np.random.seed(0)  # for reproducibility
x_random = np.linspace(-2, 2, 20)
y_random = np.random.normal(0, 5, 20)  # Mean = 0, Standard Deviation = 5

# Generate 20 points for a logarithmic curve
x_log = np.linspace(0.1, 2, 20)  # Start from 0.1 to avoid log(0)
y_log = np.log(x_log) * 10  # Multiplied by 10 to bring it to a similar scale

# Given parameters
bin_size = 0.5

# Create Curve objects for the polynomial curves
curve_A = Curve.fit(x_A4, y_A4, None, bin_size)
curve_B = Curve.fit(x_B4, y_B4, None, bin_size)
curve_C = Curve.fit(x_C4, y_C4, None, bin_size)
curve_D = Curve.fit(x_D4, y_D4, None, bin_size)

# Create a Curve object for the random curve
# Note: Using the 4th-degree polynomial fit for the random curve too, for comparison purposes
curve_random = Curve.fit(x_random, y_random, None, bin_size, ignore=True)

# Create a Curve object for the logarithmic curve
# Note: Using the 4th-degree polynomial fit for the logarithmic curve too, for comparison purposes
curve_log = Curve.fit(x_log, y_log, None, bin_size, ignore=True)

# Plot the curves alongside the original data
plt.figure(figsize=(12, 8))

# Original data
plt.scatter(x_A4, y_A4, label='Original Curve A4', c='blue', marker='o')
plt.scatter(x_B4, y_B4, label='Original Curve B4', c='green', marker='o')
plt.scatter(x_C4, y_C4, label='Original Curve C4', c='red', marker='o')
plt.scatter(x_D4, y_D4, label='Original Curve D4', c='purple', marker='o')

# Curve class data
plt.scatter(curve_A.x, curve_A.y, label='Fitted Curve A4', c='blue', marker='x')
plt.scatter(curve_B.x, curve_B.y, label='Fitted Curve B4', c='green', marker='x')
plt.scatter(curve_C.x, curve_C.y, label='Fitted Curve C4', c='red', marker='x')
plt.scatter(curve_D.x, curve_D.y, label='Fitted Curve D4', c='purple', marker='x')

# Original random data
plt.scatter(x_random, y_random, label='Original Random Curve', c='black', marker='o')

# Fitted random data
plt.scatter(curve_random.t, curve_random.dEdt, label='Fitted Random Curve', c='black', marker='x')

# Original logarithmic data
plt.scatter(x_log, y_log, label='Original Log Curve', c='orange', marker='o')

# Fitted logarithmic data
plt.scatter(curve_log.t, curve_log.dEdt, label='Fitted Log Curve', c='orange', marker='x')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Original and Fitted Fourth Degree Polynomial Curves')
plt.legend()
plt.grid(True)
plt.show()

# A and B (identical)
print(curve_A.compare(curve_B, bin_size))

# A and C (similar)
print(curve_A.compare(curve_C, bin_size))

# A and D (dissimilar)
print(curve_A.compare(curve_D, bin_size))

# A and Random (?)
print(curve_A.compare(curve_random, bin_size))

# A and Log (dissimilar)
print(curve_A.compare(curve_log, bin_size))


