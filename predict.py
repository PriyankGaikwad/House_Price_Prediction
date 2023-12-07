from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from opencage.geocoder import OpenCageGeocode

app = Flask(__name__)

rf_model = joblib.load('xgboost_model.joblib')  

opencage_api_key = 'b6472a4359a7451089a3affd5ffbb8db'

def preprocess_input(user_input):
    label_encoder = LabelEncoder()
    user_input['BHK_OR_RK'] = label_encoder.fit_transform(user_input['BHK_OR_RK'])
    user_input['POSTED_BY'] = label_encoder.fit_transform(user_input['POSTED_BY'])

    user_input = pd.get_dummies(user_input, columns=['BHK_OR_RK', 'POSTED_BY'], drop_first=True)

    expected_features_order = [
        'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE',
        'RESALE', 'LONGITUDE', 'LATITUDE', 'BHK_OR_RK_1', 'POSTED_BY_1', 'POSTED_BY_2'
    ]  
    missing_columns = set(expected_features_order) - set(user_input.columns)
    for col in missing_columns:
        user_input[col] = 0

    user_input = user_input[expected_features_order]

    return user_input

def get_lat_long(location):
    geocoder = OpenCageGeocode(opencage_api_key)
    result = geocoder.geocode(location)
    if result and len(result):
        lat = result[0]['geometry']['lat']
        lon = result[0]['geometry']['lng']
        return lat, lon
    else:
        return None, None

def predict_house_price(model, user_input):
    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_processed = preprocess_input(user_input_df)

    latitude = user_input['LATITUDE']
    longitude = user_input['LONGITUDE']

    formatted_latitude = '{:.2f}'.format(latitude)
    formatted_longitude = '{:.2f}'.format(longitude)

    print(f"User Location: {formatted_longitude}, {formatted_latitude}")

    predicted_price = model.predict(user_input_processed)

    return predicted_price[0]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_location = request.form['location']
        latitude, longitude = get_lat_long(user_location)
        
        user_input_data = {
            'UNDER_CONSTRUCTION': 0 if request.form['under_construction'] == 'yes' else 1,
            'RERA': 1 if request.form['rera'] == 'yes' else 0,
            'BHK_NO.': int(request.form['bhk_no']),
            'SQUARE_FT': float(request.form['square_ft']),
            'READY_TO_MOVE': 1 if request.form['ready_to_move'] == 'yes' else 0,
            'RESALE': 1 if request.form['resale'] == 'yes' else 0,
            'LONGITUDE': latitude,
            'LATITUDE': longitude,
            'BHK_OR_RK': request.form['bhk_or_rk'],
            'POSTED_BY': 0 if request.form['under_construction'] == 'Dealer' else 1,
        }

        predicted_price = predict_house_price(rf_model, user_input_data)
        return render_template('predict1.html', predicted_price=abs(predicted_price))

if __name__ == '__main__':
    app.run(debug=True)
