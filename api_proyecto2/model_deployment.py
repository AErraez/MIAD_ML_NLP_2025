#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def predict_api(title,plot):
    def preprocess_text(text):
        # Eliminar caracteres especiales y convertir a minúsculas
        text = re.sub(r'\W', ' ', text)
        text = text.lower()

        # Eliminar stopwords y lematizar
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text
    log_clf = joblib.load(os.path.dirname(__file__) + '/movies_log.pkl')

    vect = joblib.load(os.path.dirname(__file__) + '/movies_vect.pkl')  

    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    x_ = pd.DataFrame([[title,plot]], columns=['title','plot'])
  
    x_['plot_2'] = x_['title'] + ", " + x_['plot']
    x_['plot_2'] = x_['plot_2'].apply(preprocess_text)


    vect_x_=vect.transform(x_['plot_2'].to_list())

    # Make prediction
    p = log_clf.predict_proba(vect_x_)

    gen_p = {cols[i]: p[0][i] for i in range(len(cols))}

    return gen_p

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print('Please add all values')
        
    else:
        title = sys.argv[1]
        plot = sys.argv[2]
        p1 = predict_api(title, plot)
        print('Predicted probabilities: ', p1)