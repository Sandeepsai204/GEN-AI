import pandas as pd

dataFrame = pd.read_csv("data/Sample.csv")

dataFrame['New Global Revenue'] = dataFrame['Global Revenue'] + 2000
print(dataFrame['New Global Revenue'])