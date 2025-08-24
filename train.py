import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from joblib import dump

# Load the dataset
df = pd.read_csv('data.csv')

# Drop 'sno' column as it's an identifier
df = df.drop('sno', axis=1)

# Separate features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Encode the target variable ('yes'/'no' to 1/0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Identify categorical and numerical features
categorical_features = ['gender']
numerical_features = X.columns.drop(categorical_features).tolist()

# Create preprocessing pipelines for numerical and categorical features
# Impute missing numerical values with the median
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Preprocessing for categorical features (impute and then encode)
# In this case, we'll just handle the 'gender' column.
# For simplicity, we will encode 'gender' manually before creating the final pipeline.
X['gender'] = X['gender'].map({'male': 1, 'female': 0})

# Impute the remaining numerical columns that have NaNs
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Define and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the trained model and the imputer
dump(model, 'heart_disease_model.joblib')
dump(imputer, 'imputer.joblib')
dump(le, 'label_encoder.joblib') # Save the label encoder for prediction mapping

print("Model and preprocessing artifacts saved successfully.")
training_columns = X_imputed.columns.tolist()
dump(training_columns, 'training_columns.joblib')

print("Model, artifacts, and column order saved successfully.")
