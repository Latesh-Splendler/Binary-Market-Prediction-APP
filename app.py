import streamlit as st
import pickle
import numpy as np
import time
from sklearn.preprocessing import StandardScaler


loaded_model = pickle.load(open('C:/Users/PC/Desktop/Streamlit APP/model.pkl', 'rb'))


scaler = StandardScaler()


st.title("Binary Market Prediction App")


st.header("Enter Trade Details")



num_trades = st.number_input("Number of Trades", min_value=1, step=1)
seconds = st.number_input("Trade Duration (seconds)", min_value=1, step=1)
direction = st.selectbox("Trade Direction", options=["Rise", "Fall"])


if st.button("Predict"):
    
    input_data = scaler.fit_transform(np.array([[buy_price]]))

    
    prediction = loaded_model.predict(input_data)


    result = "Profit" if prediction[0] == 1 else "Loss"
    st.subheader(f"Predicted Outcome: {result}")


if st.button("Trade"):
    st.write(f"Initiating {num_trades} trade(s) with direction '{direction}' for {seconds} seconds each.")
    
    
    st.write("Countdown to trade execution:")
    countdown_placeholder = st.empty()  

    for i in range(seconds, 0, -1):
        countdown_placeholder.write(f"Time remaining: {i} seconds")
        time.sleep(1)  

    countdown_placeholder.write("Trade executed!")  
#python -m streamlit run app.py
