from flask import request, jsonify, render_template, Flask
import pickle
import numpy as np

app = Flask(__name__)

# Import Ridge regressor model and StandardScaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/", methods=['GET', 'POST'])
def predict_data():
    prediction_output = None
    if request.method == 'POST':
        try:
            # Collect form data  'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI'
            Temperature = float(request.form.get('Tempreture'))  # Ensure correct key name from form  
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))

            # Scale the input data
            new_data_scaled = standard_scaler.transform(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI]]
            )
            # Predict the result
            prediction_output = ridge_model.predict(new_data_scaled)[0]
        except Exception as e:
            prediction_output = f"Error: {str(e)}"

    return render_template('home.html', prediction_output=prediction_output)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
