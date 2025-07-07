from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# 16 feature names
features = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
            'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
            'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']

# Binary mapping
binary_mapping = {'yes': 1, 'no': 0, 'male': 1, 'female': 0}

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    input_data = {}
    input_list = []

    if request.method == 'POST':
        try:
            for field in features:
                value = request.form[field]
                input_data[field] = value

            for field in features:
                value = input_data[field].lower().strip()
                if field == 'Age':
                    input_list.append(int(value))
                else:
                    input_list.append(binary_mapping.get(value, 0))  # fallback to 0 if key error

            result = model.predict([input_list])[0]

            # ✅ Friendly message
            prediction = "In Risk of Diabetes" if result == 1 else "Not in Risk of Diabetes"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", features=features, prediction=prediction, form_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
