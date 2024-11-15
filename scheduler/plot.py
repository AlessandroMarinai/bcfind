import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the formula
gamma = 1.0  # You can adjust this value

# Define the formula function (updated)
def lambda_p(p, gamma):
    return 2 / (1 + np.exp(-gamma * p)) - 1

# Create a range of p values, starting from 0
p_values = np.linspace(0, 10, 500)  # Now the range starts from 0 to 10

# Compute lambda_p for each p value
lambda_values = lambda_p(p_values, gamma)

# Plot the results
plt.plot(p_values, lambda_values)
plt.title("Lambda Schedule")
plt.xlabel('p')
plt.ylabel(r'$\lambda_p$')
plt.grid(True)

plt.savefig("/home/amarinai/DeepLearningThesis/BCFind-v2/bcfind/scheduler/lambda_plot.png")