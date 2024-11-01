#!/usr/bin/python3 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import mode
import pandas as pd


#load iris data
iris = load_iris()
data = iris.data
feature_names = iris.feature_names

# Convert to a DataFrame for easier manipulation
df = pd.DataFrame(data, columns=feature_names)

# Calculate mean, median, and mode for each feature
means = df.mean()
medians = df.median()
modes = df.mode().iloc[0]  # .mode() returns a DataFrame, we take the first row

# Print the calculated values
print("Means:\n", means)
print("\nMedians:\n", medians)
print("\nModes:\n", modes)

# Plotting the mean, median, and mode
x = np.arange(len(feature_names))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width, means, width, label='Mean', color='b')
bar2 = ax.bar(x, medians, width, label='Median', color='g')
bar3 = ax.bar(x + width, modes, width, label='Mode', color='r')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Features')
ax.set_ylabel('Values')
ax.set_title('Mean, Median, and Mode of Iris Dataset Features')
ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=45)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()




