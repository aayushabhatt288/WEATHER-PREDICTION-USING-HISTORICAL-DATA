import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.title("Weather Prediction using Historical Data")

# Load dataset
data = pd.read_csv("weather.csv")

st.subheader("Dataset Preview")
st.write(data.head())

# Define inputs and target
X = data[['MinTemp', 'MaxTemp', 'Humidity', 'Pressure',
          'WindSpeed', 'CloudCover', 'Visibility',
          'Sunrise', 'Sunset']]
y = data['Temperature']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# User input section (ONLY 10 inputs)
st.subheader("Predict Temperature")

min_temp = st.number_input("Minimum Temperature (°C)")
max_temp = st.number_input("Maximum Temperature (°C)")
humidity = st.number_input("Humidity (%)")
pressure = st.number_input("Pressure (hPa)")
wind_speed = st.number_input("Wind Speed (km/h)")
cloud_cover = st.number_input("Cloud Cover (%)")
visibility = st.number_input("Visibility (km)")
sunrise = st.number_input("Sunrise Time (hours)")
sunset = st.number_input("Sunset Time (hours)")

if st.button("Predict Temperature"):
    input_data = [[min_temp, max_temp, humidity, pressure,
                   wind_speed, cloud_cover, visibility,
                   sunrise, sunset]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Temperature: {prediction[0]:.2f} °C")
