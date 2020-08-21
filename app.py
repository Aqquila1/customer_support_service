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

#предобработка сообщения
def processing_message(message):
    true_words = []
    words = re.split(' ', message)
    for word in words:
        m = re.search('(\w+)', word)
        if m is not None:
            good_word = m.group(0)
            true_words.append(good_word)
    stemming = PorterStemmer()
    stemmed_message_list = [stemming.stem(word) for word in true_words]
    stemmed_message = ' '.join(stemmed_message_list)
    return stemmed_message

# предикт категории
@application.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    resp = {'message':'ok'
           ,'category': -1
           }

    try:
        getData = request.get_data()
        json_params = json.loads(getData) 
        
        #напишите прогноз и верните его в ответе в параметре 'prediction'
        message = [json_params['user_message']]
        category = model.predict_proba(vec.transform(processing_message(message)).toarray()).tolist()
        resp['category'] = category

        
    except Exception as e: 
        print(e)
        resp['message'] = e
      
    response = jsonify(resp)
    
    return response

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)



