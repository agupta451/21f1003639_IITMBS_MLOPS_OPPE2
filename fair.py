import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from fairlearn.metrics import MetricFrame, selection_rate

# --- 1. Load and Preprocess Data ---
df = pd.read_csv('data.csv')

# Encode categorical features
gender_encoder = LabelEncoder()
target_encoder = LabelEncoder()
df['gender'] = gender_encoder.fit_transform(df['gender'])
df['target'] = target_encoder.fit_transform(df['target'])

# Define features and target
features = ['age', 'gender', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

X = df[features]
y = df[target]

# --- 2. Train Model ---
# Split data and handle missing values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_imputed, y_train)
y_pred = model.predict(X_test_imputed)

# --- 3. Fairness Analysis ---
# Get the sensitive feature ('gender') from the test set
# Note: 0 typically represents 'female', 1 represents 'male' after label encoding
sensitive_features = X_test['gender']

# Define the metrics to compute
fairness_metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate
}

# Use MetricFrame to group results by gender
metric_frame = MetricFrame(metrics=fairness_metrics,
                           y_true=y_test,
                           y_pred=y_pred,
                           sensitive_features=sensitive_features)

# --- 4. Print Results ---
print("Fairness Analysis Results by Gender:")
print(metric_frame.by_group)
