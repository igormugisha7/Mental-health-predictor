
# Mental Health Prediction in Tech Workers (Capstone Project)

# Step 1: Data Loading and Cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset (assumes CSV downloaded from Kaggle)
df = pd.read_csv("survey.csv")

# Filter valid age range
df = df[(df['Age'] > 15) & (df['Age'] < 80)]

# Clean Gender entries
def clean_gender(g):
    g = str(g).lower()
    if "male" in g or g == "m":
        return "Male"
    elif "female" in g or g == "f":
        return "Female"
    elif "non" in g:
        return "Non-binary"
    else:
        return "Other"

df['Gender'] = df['Gender'].apply(clean_gender)

# Fill missing categorical data with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Step 2: Feature Engineering
# Encode categorical variables
label_cols = ['Gender', 'self_employed', 'family_history', 'work_interfere', 'no_employees',
              'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program',
              'seek_help', 'mental_health_consequence', 'phys_health_consequence',
              'coworkers', 'supervisor', 'mental_health_interview', 'mental_vs_physical',
              'obs_consequence']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df[['Age', 'Gender', 'family_history', 'work_interfere', 'no_employees', 'remote_work',
        'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help',
        'mental_health_consequence', 'phys_health_consequence']]
y = df['treatment']  # Target column in real dataset

# Step 3: Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh')
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Step 4: Save to GitHub
# Upload this script and README.md to a GitHub repo: https://github.com/yourusername/mental-health-predictor

# Step 5: Power BI
# Export cleaned dataset as CSV and import into Power BI for dashboard creation:
df.to_csv("cleaned_mental_health.csv", index=False)
