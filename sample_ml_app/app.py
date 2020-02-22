import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model_gb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    bronx = 1
    brooklyn = 0
    manhattan = 0
    staten_island = 0

    if request.form.values("neighborhood") == 2:
        bronx = 0
        brooklyn = 1
        manhattan = 0
        staten_island = 0
    elif request.form.values("neighborhood") == 3:
        bronx = 0
        brooklyn = 0
        manhattan = 1
        staten_island = 0
    else:
        bronx = 0
        brooklyn = 0
        manhattan = 0
        staten_island = 1


    entire_home = 1
    hotel_room = 0
    private_room = 0
    shared_room = 0

    if request.form.values("room_type") == 2:
        entire_home = 0
        hotel_room = 1
        private_room = 0
        shared_room = 0
    elif request.form.values("room_type") == 3:
        entire_home = 0
        hotel_room = 0
        private_room = 1
        shared_room = 0
    else:
        entire_home = 0
        hotel_room = 0
        private_room = 0
        shared_room = 1

    int_features = [int(request.form.values("num_nights")),
                    int(request.form.values("availability")),
                    int(request.form.values("num_reviews")),
                    bronx,
                    brooklyn,
                    manhattan,
                    staten_island,
                    entire_home,
                    hotel_room,
                    private_room,
                    shared_room]

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_value=output)

if __name__ == "__main__":
    app.run(debug=True)
    app.run(host='localhost', port=5000)