import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(-1,1)
    prediction = model.predict(features)
    return render_template('index.html',prediction_text = 'Employee Salary be {}'.format(round(prediction[0][0],0)))

if __name__ == '__main__': 
    app.run(debug=True)