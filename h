from flask import Flask, request, render_template_string
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)

# -----------------------
# DATA + MODEL TRAINING
# -----------------------
data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------
# FRONTEND (HTML INSIDE PYTHON)
# -----------------------
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Predictor</title>
</head>
<body style="font-family:Arial;text-align:center;">

<h1>❤️ Heart Disease Prediction</h1>

<form action="/predict" method="post">

Age: <input name="age"><br><br>
Sex: <input name="sex"><br><br>
CP: <input name="cp"><br><br>
BP: <input name="trestbps"><br><br>
Chol: <input name="chol"><br><br>
FBS: <input name="fbs"><br><br>
ECG: <input name="restecg"><br><br>
Heart Rate: <input name="thalach"><br><br>
Exang: <input name="exang"><br><br>
Oldpeak: <input name="oldpeak"><br><br>
Slope: <input name="slope"><br><br>
CA: <input name="ca"><br><br>
Thal: <input name="thal"><br><br>

<button type="submit">Check</button>

</form>

<h2>{{result}}</h2>

</body>
</html>
"""

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def home():
    return render_template_string(html, result="")

@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"]),
    ]

    prediction = model.predict([values])[0]

    result = "⚠️ High Risk of Heart Disease" if prediction == 1 else "✅ No Risk"

    return render_template_string(html, result=result)

# -----------------------
# RUN APP
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
