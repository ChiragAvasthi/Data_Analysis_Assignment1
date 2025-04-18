from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model, feature_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_df = pd.DataFrame([data])

        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='ignore')

        input_df = pd.get_dummies(input_df)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]
        return render_template("index.html", prediction_text=f"Predicted Price: ₹{int(prediction):,}")
    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
