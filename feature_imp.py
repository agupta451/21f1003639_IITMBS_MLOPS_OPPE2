import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# --- 1. Load and Preprocess Data ---
# Load the dataset from the specified path
df = pd.read_csv('data.csv')

# Encode the 'gender' and 'target' columns into numerical format
gender_encoder = LabelEncoder()
target_encoder = LabelEncoder()
df['gender'] = gender_encoder.fit_transform(df['gender'])
df['target'] = target_encoder.fit_transform(df['target'])

# Define the feature set for the model
features = [
    'age', 'gender', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
X = df[features]
y = df['target']

# Impute missing values using the median strategy, which is robust to outliers
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.3, random_state=42
)

# --- 2. Train the Random Forest Model ---
# Initialize and train the classifier. A random_state is used for reproducibility.
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- 3. Perform SHAP Explainability Analysis ---
# Initialize the SHAP explainer with the trained model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test set. This shows how each feature contributes to each prediction.
shap_values = explainer.shap_values(X_test)

# We focus on the explanation for the "positive" class (has heart disease), which corresponds to index 1
# Take the mean of the absolute SHAP values for each feature to get a global measure of importance
mean_abs_shap = abs(shap_values[1]).mean(axis=0)

# --- 4. Rank and Display Feature Importances ---
# Create a dictionary mapping feature names to their importance scores
feature_importance = dict(zip(features, mean_abs_shap))

# Sort the features by their importance in descending order
feature_importance_sorted = sorted(
    feature_importance.items(), key=lambda x: x[1], reverse=True
)

# Print the ranked list of the most influential features
print("Top 7 Most Influential Features for Predicting Heart Disease:")
for feature, importance in feature_importance_sorted[:7]:
    print(f"- {feature}: {importance:.4f}")

