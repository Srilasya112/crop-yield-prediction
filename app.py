
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import uuid
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Paths
MODEL_PATH = 'model/crop_model.pkl'
DATA_PATH = 'model/crop_data.csv'

@app.route('/')
def home():
    return render_template('index.html')

def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        return False

    df = pd.read_csv(DATA_PATH, sep='\t')
    df.columns = df.columns.str.strip()

    # Handle missing values
    df.fillna({
        'Annual_Rainfall': df['Annual_Rainfall'].median(),
        'Fertilizer': df['Fertilizer'].median(),
        'Pesticide': df['Pesticide'].median()
    }, inplace=True)

    # Label encoding
    le_season = LabelEncoder()
    le_state = LabelEncoder()
    df['Season'] = le_season.fit_transform(df['Season'].astype(str))
    df['State'] = le_state.fit_transform(df['State'].astype(str))

    le_crop = LabelEncoder()
    df['Crop'] = le_crop.fit_transform(df['Crop'])

    joblib.dump(le_crop, 'model/crop_encoder.pkl')
    joblib.dump(le_season, 'model/season_encoder.pkl')
    joblib.dump(le_state, 'model/state_encoder.pkl')

    X = df.drop(['Crop'], axis=1)
    y = df['Crop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔍 Model Accuracy: {acc * 100:.2f}%\n")
    joblib.dump(model, MODEL_PATH)
    return True

# Train model if not already saved
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# Load trained model
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Handle image upload (optional)
        file = request.files.get('crop_image')
        uploaded_image_url = None
        if file and file.filename:
            filename = str(uuid.uuid4()) + "_" + file.filename
            upload_path = os.path.join("static/uploads", filename)
            file.save(upload_path)
            uploaded_image_url = url_for('static', filename='uploads/' + filename)

        # Prepare input data
        input_data = pd.DataFrame([{
            'Crop_Year': int(data['year']),
            'Season': data['season'],
            'State': data['state'],
            'Area': float(data['area']),
            'Production': float(data['production']),
            'Annual_Rainfall': float(data['rainfall']),
            'Fertilizer': float(data['fertilizer']),
            'Pesticide': float(data['pesticide']),
            'Yield': float(data['yield'])
        }])

        # Manual encoding (must match training encoders)
        season_map = {'Kharif': 0, 'Rabi': 1, 'Summer': 2, 'Winter': 3}
        state_map = {'Andhra Pradesh': 0, 'Telangana': 1, 'Maharashtra': 2, 'Punjab': 3, 'Haryana': 4}
        input_data['Season'] = input_data['Season'].map(season_map)
        input_data['State'] = input_data['State'].map(state_map)

        # 🔧 Add engineered features
        input_data['Rainfall_per_Production'] = input_data['Annual_Rainfall'] / (input_data['Production'] + 1)
        input_data['Fert_Pest_Ratio'] = input_data['Fertilizer'] / (input_data['Pesticide'] + 1)
        input_data['Area_Production_Ratio'] = input_data['Area'] / (input_data['Production'] + 1)

        # Make prediction
        encoded_prediction = model.predict(input_data)[0]
        crop_encoder = joblib.load('model/crop_encoder.pkl')
        prediction = crop_encoder.inverse_transform([encoded_prediction])[0]

        # Image logic
        crop_key = prediction.lower().replace(" ", "_").replace("-", "_")
        image_name_map = {
            "lady_finger": "ladysfinger",
            "soyabean": "soya_bean",
            "groundnut": "ground_nut",
            "sugarcane": "sugar_cane",
        }
        crop_key = image_name_map.get(crop_key, crop_key)
        image_filename = f'images/{crop_key}.jpg'
        image_path = os.path.join('static', image_filename)

        image_url = url_for('static', filename=image_filename if os.path.exists(image_path) else 'images/no_image.jpg')

        return render_template('result.html',
                               prediction=prediction,
                               input_data=data,
                               image_url=image_url,
                               uploaded_image_url=uploaded_image_url)

    except Exception as e:
        return jsonify({'error': str(e)}), 400



# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# File checker (optional)
def static_file_exists(filename):
    static_path = os.path.join(app.static_folder, filename)
    return os.path.isfile(static_path)

@app.context_processor
def utility_processor():
    return dict(static_files=static_file_exists)

if __name__ == '__main__':
    app.run(debug=True)
