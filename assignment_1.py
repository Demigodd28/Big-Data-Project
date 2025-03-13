import numpy as np
import matplotlib.pyplot as plt

# Generate 100 samples
np.random.seed(0)
size = np.random.uniform(5, 10, 100)
price = np.exp(size) * 0.001 + np.random.uniform(-0.6, 0.6, 100)  # Price correlated with size + noise

# Plot the data
plt.scatter(size, price, color='blue', alpha=0.5, s=20)

# Plot exp
x = np.linspace(5, 10, 100)  # Generate smooth x values
y = np.exp(x) * 0.001  # Compute the original function
plt.plot(x, y, color='red', linestyle='dashed', linewidth=2, label="Original sin(x)")

plt.xlabel("Size (per ten sq meters)")
plt.ylabel("Price (ten millions of $)")
plt.title("House Price vs Size")
plt.grid(True)
plt.show()
