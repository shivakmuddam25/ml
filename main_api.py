# from fastapi import FastAPI
# app = FastAPI()

# @app.get("/")
# async def func(x: int, y: int):
#     total = x + y
#     return {"Sum": total}
# import subprocess
# subprocess.run("uvicorn test1:app --reload", shell=True)


## Import Libraries
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from huggingface_hub import hf_hub_download
import joblib
from fastapi import FastAPI, Query, status
from typing import Optional
import uvicorn
app = FastAPI()

# Initialize the variables
REPO_ID = "Shivakumar25/charrec"
FILENAME1 = "rf_amazon_rev_bow.sav"
FILENAME2 = "rf_amazon_rev.sav"
stop_words = stopwords.words("english") + ["br", "html", "www", "k", "http"]
# rf_bow_amazon_rev = pickle.load(open("rf_amazon_rev_bow.sav", "rb"))
# rf_amazon_rev = pickle.load(open("rf_amazon_rev.sav", "rb"))
rf_bow_amazon_rev = joblib.load(hf_hub_download(REPO_ID, filename=FILENAME1))
rf_amazon_rev = joblib.load(hf_hub_download(REPO_ID, filename=FILENAME2))

# st.subheader("Predict the Amazon Review Score")
# review = st.text_input("Review Summary:")
# review_desc = st.text_area("Review Description:")

class Data(BaseModel):
     review: str
     review_desc : str

@app.post("/predict")
async def predict_review(data: Data):
    data_dict = data.dict()
    final_text = data_dict['review'] + " " + data_dict['review_desc']
    if int(len(final_text)) == 1:
        # st.write("Please input the Review")
        pass
    else:           
            final_text = list(map(lambda x:x.replace(":)", "good "),  final_text.split(" ")))
            final_text = list(map(lambda x:x.replace(":(", "bad "),  final_text))
            final_text_re = re.sub("[^a-zA-Z]", " ", str(final_text).lower())
            lemmatize = WordNetLemmatizer() # stemmer = PorterStemmer()
            lem_text = [lemmatize.lemmatize(x.lower()) for x in final_text_re.split(" ") if x not in stop_words]
            # print("1.", lem_text)
            lem_text = [lemmatize.lemmatize(x) for x in lem_text if x!=""]
            lem_text = " ".join([x for x in lem_text])
            # print("2.", lem_text)
            res = lambda lem_text: ' '.join(set(row_word for row_word in lem_text.split(" ") if row_word not in stop_words))
            lem_text = res(lem_text)
            # print("3.", lem_text)
            X_test_ = rf_bow_amazon_rev.transform([lem_text])  
            # print("4.", X_test_)
            score = rf_amazon_rev.predict(X_test_.toarray().reshape(1, -1))
            # print("5.", score[0])
            return {"Predicted Score": int(score[0])}
    
if __name__ == '__main__':
     uvicorn.run("main_api:app", host="127.0.0.1", port =8000, reload = True)