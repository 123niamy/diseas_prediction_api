from sklearn.datasets import make_classification
import pandas as pd

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,      # number of samples
    n_features=10,       # total features
    n_informative=5,     # number of useful features
    n_redundant=2,       # number of redundant features
    n_classes=2,         # binary classification
    random_state=42
)

# Convert to DataFrame for readability
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df["target"] = y

print(df.head())
