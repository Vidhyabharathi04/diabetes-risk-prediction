import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("early_diabetes_data.csv")  # make sure the file is in the same folder

# 2. Clean text inputs
df = df.applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# 3. Replace binary strings with 1s and 0s
binary_mapping = {
    'yes': 1,
    'no': 0,
    'male': 1,
    'female': 0,
    'positive': 1,
    'negative': 0
}
df = df.replace(binary_mapping)

# 4. Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# 5. Drop missing values
print("Before dropping NA:", len(df))
df.dropna(inplace=True)
print("After dropping NA:", len(df))

# 6. Define features and target
X = df.drop("class", axis=1)
y = df["class"]

# 7. SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 8. Scale features (only for Logistic Regression)
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# 9. Setup Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 10. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 11. Cross-validation evaluation
print("\n📊 Cross-Validation Results (on SMOTE-resampled data):")
for name, model in models.items():
    if name == "Logistic Regression":
        scores = cross_val_score(model, X_resampled_scaled, y_resampled, cv=skf, scoring='accuracy')
    else:
        scores = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='accuracy')
    
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")

# 12. Optional: Train best model on full resampled data and evaluate on a real test split

# Split again for final test evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE only to training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train best model (Random Forest)
best_model = RandomForestClassifier()
best_model.fit(X_train_resampled, y_train_resampled)
y_pred = best_model.predict(X_test)
import joblib

# Save the trained model to a file
joblib.dump(best_model, 'diabetes_model.pkl')
print("✅ Model saved as 'diabetes_model.pkl'")


# Metrics and confusion matrix
print("\n📌 Final Evaluation on Held-Out Test Set:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest (Final Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("final_confusion_matrix.png")
print("✅ Final confusion matrix saved as 'final_confusion_matrix.png'")
# Bar graph: model accuracy comparison
import matplotlib.pyplot as plt

# Accuracy values from cross-validation
model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
accuracies = [0.9297, 0.9844, 0.9734]  # Replace with your actual printed values

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies, color=['skyblue', 'green', 'orange'])
plt.ylim(0.85, 1.0)
plt.ylabel("Cross-Validated Accuracy")
plt.title("Comparison of Model Accuracies")

# Add value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()
print("📊 Accuracy comparison chart saved as 'model_accuracy_comparison.png'")
import joblib

# Save the model
joblib.dump(rf_model, 'diabetes_model.pkl')
