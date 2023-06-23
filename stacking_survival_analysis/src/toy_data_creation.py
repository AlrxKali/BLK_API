import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate random values for each column
feature_1 = np.random.randint(1, 26, size=150)
feature_2 = np.random.randint(26, 50, size=150)
feature_3 = np.random.randint(0, 2, size=150)
feature_4 = np.random.choice(np.arange(1, 13) * 5, size=150)

# Create a dictionary with column names and values
data = {
    'feature_1': feature_1,
    'feature_2': feature_2,
    'event': feature_3,
    'time': feature_4
}

# Create a pandas DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('../stacking_survival_analysis/data/data.csv', index=False)