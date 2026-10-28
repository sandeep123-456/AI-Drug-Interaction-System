import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

print("Starting Model Training Pipeline...")

# 1. Load Dataset
data_path = 'drug_interactions.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found in the current directory.")

df = pd.read_csv(data_path)

# 2. Preprocessing
severity_map = {'Low': 0, 'Moderate': 1, 'High': 2}
df['Severity_Target'] = df['Severity'].map(severity_map)

# Encode Drug Names
encoder = LabelEncoder()
all_drugs = pd.concat([df['Drug_A'], df['Drug_B']]).unique()
encoder.fit(all_drugs)

df['Drug_A_Encoded'] = encoder.transform(df['Drug_A'])
df['Drug_B_Encoded'] = encoder.transform(df['Drug_B'])

X = df[['Drug_A_Encoded', 'Drug_B_Encoded']]
y = df['Severity_Target']

# 3. Model Training
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)

# 4. Save Artifacts (Model, Encoder, and Map)
os.makedirs('saved_models', exist_ok=True)

with open('saved_models/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('saved_models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('saved_models/severity_map.pkl', 'wb') as f:
    pickle.dump(severity_map, f)

print("✅ Training Complete! Model and Encoder saved in 'saved_models/' directory.")