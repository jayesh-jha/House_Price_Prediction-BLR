import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title='My Webpage', page_icon=":tada:", layout="wide")

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

def load_lottie_url(url):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	return r.json()


#---------- LOAD ASSET -----------
lottie_obj = load_lottie_url("https://lottie.host/05f3b9af-813c-4e81-ad54-c0bc2b8e9619/wY5vxYUxrJ.json")

#--------- Import pre-trained model --------------
with open(r'data/pipe_ridge.pkl', 'rb') as f:
    pipe = pickle.load(f) # deserialize using load()


data = pd.read_csv(r"data/filtered_data.csv")
locations = data['location'].unique()
locations = np.insert(locations, 0, "--Select--", axis=None)





#------ Header Section -------
with st.container():
	st.title("House Price Prediction Model!!")
	# st.write("a machine learning model, that uses Linear Regression to predict House Prices.")
	st.write("This [dataset](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) was used to train the model !")

location, bath, area, bhk = None, None, None, None
with st.container():
	left_form, right_animation = st.columns((2,1))

	with left_form:
		with st.form("my_form"):

			location = st.selectbox('Location:', locations)
			st.caption("if you cannot find your desired location, select the option \"Other\".")

			inp_bath = st.empty()
			bath = (inp_bath.text_input("No. of Bathrooms:"))


			inp_area = st.empty()
			area = (inp_area.text_input("Total Area in Sq. Ft.:"))

			inp_bhk = st.empty()
			bhk = (inp_bhk.text_input("Desired no. of bedrooms (BHK):"))

			submitted = st.form_submit_button("Predict Price")

	with right_animation:
		st_lottie(lottie_obj, height = 300)


try:
	if area and bath and bhk:
		area, bhk, bath = float(area), float(bhk), float(bath)

		user_input = pd.DataFrame([[location, area, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
		prediction = round(float(pipe.predict(user_input)[0])*100000, 2) 		#------------------ since the data was stored in laks, multiplying by 1 lakh

		if prediction < 20000:
			raise Exception

		st.header(f"₹ {prediction}")

	elif submitted:
		st.write("**Invalid Input !!**")
except:
	st.write("**Invalid Input !!**")