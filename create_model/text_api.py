"""
Created on 2020-12-09
@author: Dr. Ganjar Alfian
email : ganjar@dongguk.edu
for teaching purpose only.
"""

import pandas as pd
import joblib
from flask import Flask, redirect, url_for, request, render_template
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import urllib.request as request_api
import json

#from waitress import serve
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)
@app.route('/sentence/prediction/', methods=["POST"])
def prediction_result():
    #receive parameter sent by client
    sentence = request.form.get('sentence')
    #sentence = request.args.get('sentence')

    #start making prediction
    transformer = TfidfTransformer()
    #load the vectorizer 
    vector_model = 'vectorizer.model'
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=joblib.load(open(vector_model, "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array([sentence])))
    #load the model
    filename = 'mlp_tfidf.model'
    loaded_model= joblib.load(filename)
    #make new prediction
    result = loaded_model.predict(tfidf)
    #print(result)
    
    for i in result:
        int_result = int(i)
        if(int_result==0):
            decision='No Stress'
        else:
            decision='Stress'
    
    
    dictionary = {
            'status': 'success',
            'prediction': decision
    }
    jsonDataOutput=json.dumps(dictionary)
    return jsonDataOutput


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port='5000')
    app.run()
    #serve(app, host='127.0.0.1', port=5000)
    