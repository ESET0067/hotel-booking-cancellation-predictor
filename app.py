import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# -----------------------------
# Load model and preprocessor
# -----------------------------
model = keras.models.load_model("hotel_model.keras")
preprocessor = joblib.load("preprocessor.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Enter the booking details below to predict whether the booking is likely to be **canceled**.")

# -----------------------------
# Input fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
    lead_time = st.number_input("Lead Time (days before arrival)", min_value=0, max_value=500, value=50)
    arrival_date_month = st.selectbox("Arrival Month", list(range(1, 13)), format_func=lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
    arrival_date_week_number = st.number_input("Arrival Week Number", 1, 52, 20)
    arrival_date_day_of_month = st.number_input("Arrival Day of Month", 1, 31, 15)
    stays_in_weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
    stays_in_week_nights = st.number_input("Week Nights", 0, 20, 2)
    adults = st.number_input("Number of Adults", 1, 10, 2)
    children = st.number_input("Number of Children", 0, 10, 0)
    babies = st.number_input("Number of Babies", 0, 5, 0)

with col2:
    is_repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
    previous_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)
    previous_bookings_not_canceled = st.number_input("Previous Non-Canceled Bookings", 0, 10, 0)
    required_car_parking_spaces = st.number_input("Car Parking Spaces", 0, 5, 0)
    total_of_special_requests = st.number_input("Special Requests", 0, 5, 0)
    adr = st.number_input("Average Daily Rate (€)", 0.0, 1000.0, 100.0)
    meal = st.selectbox("Meal Type", ["BB", "FB", "HB", "SC"])
    market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Complementary"])
    distribution_channel = st.selectbox("Distribution Channel", ["TA/TO", "Direct", "Corporate", "GDS"])
    reserved_room_type = st.selectbox("Reserved Room Type", list("ABCDEFGH"))
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Group", "Transient-Party"])

# -----------------------------
# Collect all input into DataFrame
# -----------------------------
input_dict = {
    "hotel": [hotel],
    "lead_time": [lead_time],
    "arrival_date_month": [arrival_date_month],
    "arrival_date_week_number": [arrival_date_week_number],
    "arrival_date_day_of_month": [arrival_date_day_of_month],
    "stays_in_weekend_nights": [stays_in_weekend_nights],
    "stays_in_week_nights": [stays_in_week_nights],
    "adults": [adults],
    "children": [children],
    "babies": [babies],
    "is_repeated_guest": [is_repeated_guest],
    "previous_cancellations": [previous_cancellations],
    "previous_bookings_not_canceled": [previous_bookings_not_canceled],
    "required_car_parking_spaces": [required_car_parking_spaces],
    "total_of_special_requests": [total_of_special_requests],
    "adr": [adr],
    "meal": [meal],
    "market_segment": [market_segment],
    "distribution_channel": [distribution_channel],
    "reserved_room_type": [reserved_room_type],
    "deposit_type": [deposit_type],
    "customer_type": [customer_type],
}

input_df = pd.DataFrame.from_dict(input_dict)

# -----------------------------
# Preprocess the input
# -----------------------------
input_processed = preprocessor.transform(input_df)

# -----------------------------
# Make prediction
# -----------------------------
if st.button("🔮 Predict Cancellation"):
    prediction = model.predict(input_processed)[0][0]
    probability = float(prediction) * 100

    if prediction >= 0.5:
        st.error(f"❌ Likely to be CANCELED ({probability:.2f}% probability)")
    else:
        st.success(f"✅ Likely to be NOT CANCELED ({100 - probability:.2f}% probability)")
