# flask  ,scikit-learn , pandas , pickle-mixin
import pandas as pd
import pickle
import numpy as np
from os import pipe



from flask import Flask, request, render_template

app = Flask(__name__)

data = pd.read_csv('Cleaned_data.csv')
model = pickle.load(open('RidgeModel.pkl', 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])



def predict():
    location = request.form.get('location')
    bhk = request.form.get('BHK')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    prediction = model.predict(input)[0] * 1e5

    return str(np.round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True)
