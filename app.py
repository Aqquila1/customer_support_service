import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle



from flask import Flask
from flask import request
import requests
from flask import jsonify

import os
import json
from ast import literal_eval
import traceback
import re

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from sumy.utils import get_stop_words as gsw1
from stop_words import get_stop_words as gsw2

import pickle
from nltk.stem.porter import *
stemming = PorterStemmer()



application = Flask(__name__)


#загружаем модели из файла
vec = pickle.load(open("./models/tfidf.pickle", "rb"))
# model = lgb.Booster(model_file='./models/lgbm_model.txt')
with open("./models/mlp_model.pkl", 'rb') as file:
    model = pickle.load(file)


# тестовый вывод
@application.route("/")  
def hello():
    resp = {'message':"Hello World!"}
    
    response = jsonify(resp)
    
    return response


def cleaning_message(text):
    text = re.sub('\[.*\]', '', text)
    text = re.sub("\!", '', text)
    text = re.sub("\'", '', text)
    text = re.sub("[^A-Za-z0-9^,!.\/'+-=]", ' ', text)
    text = re.sub("\s+", ' ', text)
    return text


def get_original_form(text):
    try:
        words = re.split(' ', text)
        true_words = []
        for word in words:
            m = re.search('(\w+)', word)
            if m is not None:
                good_word = m.group(0)
                true_words.append(good_word)
        tagged = nltk.pos_tag(true_words)
        tags = []
        for tag in tagged:
            tags.append(tag[1])
    except:
        pass

    stopWords = nltk.corpus.stopwords.words()
    LANGUAGE = 'english'

    sw0 = ["yeah", "zola", "don"]
    sw1 = gsw1(LANGUAGE)
    sw2 = gsw2('en')

    sw0.extend(list(sw1))
    sw0.extend(list(sw2))

    new_s = ''
    new_lw = []
    new_lt = []
    new_lwt = []
    for w, t, lw in zip(true_words, tags, tagged):
        if t in ['NN', 'VB', 'DT', 'NNS', 'VBP', 'VB']:
            new_s += w + ' '
            new_lw.append(w)
            new_lt.append(t)
            new_lwt.append(lw)
        elif w in sw0 or re.match('\d+', w) is not None:
            continue
    new_list = re.split(r'\W+', new_s)
    stemmed_list = [stemming.stem(word) for word in new_list]
    original_form = ' '.join(stemmed_list)
    return original_form

# предикт категории
@application.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    resp = {'message':'ok'
           ,'category': -1
           }

    try:
        getData = request.get_data()
        json_params = json.loads(getData)

        message = json_params['user_message']
        message = message.lower()
        message = cleaning_message(message)
        message = get_original_form(message)
        
        #напишите прогноз и верните его в ответе в параметре 'prediction'
        prediction = model.predict_proba(vec.transform([message]).toarray()).tolist()
        resp['category'] = prediction

        
    except Exception as e: 
        print(e)
        resp['message'] = e
      
    response = jsonify(resp)
    
    return response

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)



