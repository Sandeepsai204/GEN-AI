import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataFrame = pd.read_csv("data/Sample.csv")
print(dataFrame.head())
dataFrame = pd.DataFrame(dataFrame)

# Bar Plot using Matplotlib
plt.bar(dataFrame['Date'], dataFrame['Global Revenue'])
plt.show()

# Bar Plot using Seaborn
dataFrame = sns.load_dataset("iris")
print(dataFrame.species.value_counts().plot(kind="barh"))

plt.title("Species Count", fontsize=16, fontweight='bold', color='Green')
plt.xlabel("Count")
plt.ylabel("Species")

plt.show() 