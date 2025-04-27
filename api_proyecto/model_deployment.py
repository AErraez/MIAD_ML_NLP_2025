#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_api(dance,energy,speech,loud):

    rf = joblib.load(os.path.dirname(__file__) + '/spotify_rf.pkl') 

    x_ = pd.DataFrame([[dance,energy,speech,loud]], columns=['danceability','energy','speechiness','loudness'])
  
    x_=x_.astype(float)

    # Make prediction
    p1 = rf.predict(x_)[0]

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print('Please add all values')
        
    else:

        dance = float(sys.argv[1])
        energy = float(sys.argv[2])
        speech = float(sys.argv[3])
        loud = float(sys.argv[4])

        p1 = predict_api(dance,energy,speech,loud)
        
        print('Predicted Popularity: ', p1)
        