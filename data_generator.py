import pandas as pd
import numpy as np

# Define the columns to match the training data
columns = [
    "age", "gender", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def random_data_generator(num_samples):
    """Generates a DataFrame with random patient data."""
    np.random.seed(42) # for reproducibility
    data = {
        "age": np.random.randint(29, 78, num_samples),
        "gender": np.random.choice(["male", "female"], num_samples),
        "cp": np.random.randint(0, 4, num_samples),
        "trestbps": np.random.randint(94, 201, num_samples),
        "chol": np.random.randint(126, 565, num_samples),
        "fbs": np.random.randint(0, 2, num_samples),
        "restecg": np.random.randint(0, 3, num_samples),
        "thalach": np.random.randint(71, 203, num_samples),
        "exang": np.random.randint(0, 2, num_samples),
        "oldpeak": np.round(np.random.uniform(0, 6.2, num_samples), 1),
        "slope": np.random.randint(0, 3, num_samples),
        "ca": np.random.randint(0, 5, num_samples), # Note: ca can go up to 4 in your data
        "thal": np.random.randint(0, 4, num_samples)  # Note: thal can go up to 3
    }
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate and save 100 samples
generated_df = random_data_generator(100)
generated_df.to_csv("test.csv", index=False)

print("test.csv with 100 samples created successfully.")
