import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

# 1. Generate synthetic data (replace with your real dataset loading)
np.random.seed(42)
n_samples = 1000
data = {
    'age': np.random.randint(20, 80, n_samples),
    'gender': np.random.randint(0, 2, n_samples),  # 0: female, 1: male
    'blood_pressure': np.random.randint(90, 180, n_samples),
    'cholesterol': np.random.randint(150, 300, n_samples),
    'glucose': np.random.randint(70, 200, n_samples),
    'bmi': np.random.uniform(18, 40, n_samples),
    'family_history': np.random.randint(0, 2, n_samples),  # 0: no, 1: yes
    'smoking': np.random.randint(0, 2, n_samples),
}
# Example target: 1 = has disease, 0 = no disease
data['disease'] = (
    ((data['glucose'] > 125) | (data['bmi'] > 30)) & 
    (data['family_history'] == 1)
).astype(int)

df = pd.DataFrame(data)
X = df.drop('disease', axis=1)
y = df['disease']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Handle missing data (for illustration, let's add some missing values)
X_train.iloc[0, 0] = np.nan  # Introduce a missing value
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 4. Model training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_imputed, y_train)

# 5. Evaluation
y_pred = clf.predict(X_test_imputed)
y_proba = clf.predict_proba(X_test_imputed)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Save the trained model for API integration
import joblib
joblib.dump(clf, "disease_model.joblib")