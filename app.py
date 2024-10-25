from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('lung_cancer_predictor.joblib')

@app.route('/')
def home():
    return render_template('index.html')  # Serve the homepage

@app.route('/submit', methods=['POST'])
def submit():
    # Collect data from the form
    age = int(request.form['age'])
    allergy = int(request.form['allergy'])
    peer_pressure = int(request.form['peer_pressure'])
    alcohol_consuming = int(request.form['alcohol_consuming'])

    # Create a DataFrame for the model input
    input_data = pd.DataFrame([[age, allergy, peer_pressure, alcohol_consuming]], 
                              columns=["AGE", "ALLERGY ", "PEER_PRESSURE", "ALCOHOL CONSUMING"])

    # Get the prediction
    prediction = model.predict(input_data)[0]

    # Return the result back to the webpage
    result = "lol happy having Lung Disease" if prediction == 1 else "You Dont have Lung Disease"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
