import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Sample dataset (you can replace this with a real dataset)
data = {
    'Soil pH': [6.5, 5.8, 7.2, 6.0, 5.5, 6.8, 6.3, 7.0, 6.1, 6.7],
    'Moisture': [30, 45, 50, 40, 35, 60, 55, 65, 50, 48],
    'Nitrogen': [50, 40, 55, 35, 30, 60, 45, 70, 40, 50],
    'Phosphorus': [30, 25, 40, 20, 18, 45, 30, 50, 25, 35],
    'Potassium': [25, 30, 35, 20, 15, 40, 28, 45, 30, 33],
    'Temperature': [25, 30, 28, 22, 24, 27, 26, 29, 23, 25],
    'Rainfall': [200, 250, 300, 150, 180, 270, 230, 290, 210, 260],
    'Soil Condition': ['Good', 'Moderate', 'Good', 'Poor', 'Moderate', 'Good', 'Moderate', 'Good', 'Poor', 'Good'],
    'Crop': ['Wheat', 'Rice', 'Corn', 'Rice', 'None', 'Wheat', 'Rice', 'Corn', 'None', 'Wheat']
}

df = pd.DataFrame(data)

# Encode categorical labels
label_encoder_soil = LabelEncoder()
label_encoder_crop = LabelEncoder()
df['Soil Condition'] = label_encoder_soil.fit_transform(df['Soil Condition'])
df['Crop'] = label_encoder_crop.fit_transform(df['Crop'])

# Features and target variables
X = df[['Soil pH', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Rainfall']]
y_soil = df['Soil Condition']
y_crop = df['Crop']

# Split data
X_train, X_test, y_train_soil, y_test_soil = train_test_split(X, y_soil, test_size=0.2, random_state=42, stratify=y_soil)
_, _, y_train_crop, y_test_crop = train_test_split(X, y_crop, test_size=0.2, random_state=42, stratify=y_crop)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models (handling KNN issue)
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
}

# Add KNN only if enough training samples exist
if len(X_train) >= 5:
    models['KNN'] = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))

# Train models and store best ones
best_model_soil, best_model_crop = None, None
best_acc_soil, best_acc_crop = 0, 0

for name, model in models.items():
    model.fit(X_train, y_train_soil)
    acc = accuracy_score(y_test_soil, model.predict(X_test))
    if acc > best_acc_soil:
        best_acc_soil = acc
        best_model_soil = model

    model.fit(X_train, y_train_crop)
    acc = accuracy_score(y_test_crop, model.predict(X_test))
    if acc > best_acc_crop:
        best_acc_crop = acc
        best_model_crop = model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_features = [
            float(data["soil_pH"]),
            float(data["moisture"]),
            float(data["nitrogen"]),
            float(data["phosphorus"]),
            float(data["potassium"]),
            float(data["temperature"]),
            float(data["rainfall"])
        ]
        input_scaled = scaler.transform([input_features])

        predicted_soil = best_model_soil.predict(input_scaled)
        predicted_crop = best_model_crop.predict(input_scaled)

        response = {
            "soil_condition": label_encoder_soil.inverse_transform(predicted_soil)[0],
            "crop_recommendation": label_encoder_crop.inverse_transform(predicted_crop)[0]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
