import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# --- 1. Load the Original (Clean) Data ---
df_original = pd.read_csv('data.csv')

# --- 2. Create the Poisoned Dataset ---
# An attacker swaps the target labels
df_poisoned = df_original.copy()
df_poisoned['target'] = df_original['target'].map({'yes': 'no', 'no': 'yes'})

# --- 3. Preprocessing ---
# Define features and target
features = ['age', 'gender', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

# Handle categorical 'gender' feature
gender_encoder = LabelEncoder()
df_original['gender'] = gender_encoder.fit_transform(df_original['gender'])

# Prepare data for the model
X = df_original[features]
y_original = df_original[target]

# Also get the poisoned labels for later comparison
y_poisoned = df_poisoned[target] 

# --- 4. Train the Model on CLEAN Data ---
# Split the ORIGINAL data to create a trusted training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_original, test_size=0.3, random_state=42)

# Handle missing values using the imputer from our MLOps pipeline
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_imputed, y_train)

# --- 5. Evaluate Performance ---

# A) Performance on the original, clean test data
predictions_on_clean = model.predict(X_test_imputed)
accuracy_clean = accuracy_score(y_test, predictions_on_clean)

# B) Performance against the poisoned labels
# We use the same X_test features, but compare against the swapped labels
poisoned_test_labels = y_poisoned.loc[X_test.index]
accuracy_poisoned = accuracy_score(poisoned_test_labels, predictions_on_clean)

# --- 6. Print Results ---
print(f"Accuracy on Clean Data: {accuracy_clean:.2%}")
print(f"Accuracy against Poisoned Data: {accuracy_poisoned:.2%}")

