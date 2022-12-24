import streamlit as st
from PIL import Image
import pickle
import numpy as np
import time
from datetime import datetime
from time import gmtime, strftime
size = (10, 10)

st.subheader("Predict the Character/Alphabet")
file = st.file_uploader(label = "Input the picture Image", label_visibility="hidden")
if file is not None:
    img = Image.open(file)
    # st.image(img.resize((10, 10)), caption = "Uploaded Character Image") 
    st.image(img.resize(size), caption = "Uploaded Character Image") 

def predict_query():
    with st.spinner("Wait while we work on it....."):
        # st_time = datetime.now()
        # st.write(st_time)
        if file is not None:
            loaded_rf_model = pickle.load(open("mnist_RF_pickle_.sav", "rb"))
            # loaded_rf_model = pickle.load(open("rf_model.pkl", "rb"))
            q_img = Image.open(file)
            q_img = q_img.convert("L")
            # q_img = q_img.resize((10, 10))
            q_img = q_img.resize(size)
            q_img = np.array(q_img)/255
            q_img = q_img.ravel()        
            y_pred = loaded_rf_model.predict(q_img.reshape(1, -1))
            y_pred_proba = loaded_rf_model.predict_proba(q_img.reshape(1, -1))
            acc = np.max(y_pred_proba)*100
            if acc > 35:
                st.write("The predicted class is: {0}   ({1:.2f}% accuracy)".format(y_pred[0],np.max(y_pred_proba)*100))
            else:
                st.write("The predicted class is: {0}   ({1:.2f}% accuracy)".format(y_pred[0],np.max(y_pred_proba)*100))
                st.write("Sorry!! Could not recognize the Image. Try another.")
            # end_time = datetime.now()
            # st.write("Execution time: {0}".format(strftime("%H:%M:%S", end_time-st_time)))
        else:
            st.write("No file was chosen")
if st.button("Predict", help="Click to Predict the class!"):
     predict_query()
 