import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict",methods=['POST'])
def predict():
    try:
        ChestPainType_NAP = 0;
        ChestPainType_TA = 0;
        ChestPainType_ATA = 0;

        # Extract features from form
        chest_value = request.form['chestPainType']
        chest_value = int(chest_value)
        print(chest_value)
        if chest_value == 1:
            print("Get Atypical Angina")
            ChestPainType_ATA = 1;
        elif chest_value == 2:
            print("Get ChestPainType_NAP")
            ChestPainType_NAP = 1;
        elif chest_value == 3:
            print("Get ChestPainType_TA")
            ChestPainType_TA = 1;
        else:
            print("normal")

        RestingECG_Normal = 0;
        RestingECG_ST = 0;
        restECG_value = request.form['restingECG']
        restECG_value = int(restECG_value)
        if restECG_value == 1:
            print("Get restingECG normal")
            RestingECG_Normal = 1;
        elif restECG_value == 2:
            print("Get RestingECG_ST")
            RestingECG_ST = 1;

        ST_Slope_Flat = 0;
        ST_Slope_Up = 0;
        slope_value = request.form['stSlope']
        slope_value = int(slope_value)
        if slope_value == 1:
            print("Get stSlope flat")
            ST_Slope_Flat = 1;
        elif slope_value == 2:
            print("Get ST_Slope_Up")
            ST_Slope_Up = 1;

        int_features = [
            int(request.form['age']),
            int(request.form['restingBP']),
            int(request.form['cholesterol']),
            int(request.form['fastingBS']),
            int(request.form['maxHR']),
            float(request.form['oldpeak']),
            int(request.form['sex']),
            ChestPainType_ATA,
            ChestPainType_NAP,
            ChestPainType_TA,
            RestingECG_Normal,
            RestingECG_ST,
            request.form['exerciseAngina'],
            ST_Slope_Flat,
            ST_Slope_Up,
        ]
        features = [np.array(int_features)]

        print(features)
        # print("\n")
        features = scaler.transform(features)
        prediction = model.predict(features)
        print(prediction)

        if(prediction == 0):
            result = "Don't have heart disease"
        else:
            result = "Have heart disease"
        print(result)

        return render_template('index.html', prediction_text='Prediction: {}'.format(result))
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(port=8085,debug=True)