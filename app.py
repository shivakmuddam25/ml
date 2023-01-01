import streamlit as st
import requests
import json
import socket
import datetime

st_time = datetime.datetime.now()
# st.write(st_time)

def run():
    st.subheader("Predict the Amazon Review Score")
    review = st.text_input("Review Summary:")
    review_desc = st.text_area("Review Description:")
    if st.button("Predict"):
        data = {
            "review": review,
            "review_desc": review_desc
        }
        # API_URL = "aws-charrec-4643" 
        API_URL = r"http://{0}:8000".format(socket.gethostname()) 
        # res = requests.post("http://127.0.0.1:8000/predict", json=data)
        res = requests.post(r"{0}/predict".format(API_URL), json=data)
        print(res.text) 
       
        # res = requests.post("POST", API_URL, json=data)      
  
        end_time = datetime.datetime.now()
        
        if res.json():
            score = res.json()["Predicted Score"]
            st.write("★ "* score, "☆ "* (5-score))
            # st.success("Predicted Score is {0}".format(res.json()["Predicted Score"]))
            st.write(res.json())
        else:
            st.warning("Please provide the Review")

        st.write("Execution Time : {0}".format(end_time-st_time))
                
if __name__ == '__main__':
    run()
