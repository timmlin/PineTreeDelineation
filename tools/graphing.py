import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv("results/layer_stacking_Results.csv")
df.columns = df.columns.str.strip()

# Plot data
plt.figure(figsize=(10, 5))
plt.bar(df['deafult_layer_stacking_treetop_count'], df['relative_error'], width = 2, color = '#4F7942', edgecolor = '#3E2723')

# Formatting the plot
plt.xlabel("predicted tree count", fontsize = 30)
plt.ylabel("exe_times (s)", fontsize = 30)
plt.title("Layer Stacking Execution Times vs. predicted Tree Count", fontsize = 36)
plt.grid(True)
plt.xticks(fontsize = 16, rotation=45)
plt.yticks(fontsize = 16)

# Show plot
plt.show()



