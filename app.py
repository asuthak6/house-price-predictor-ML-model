import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model/house_price_model.pkl')

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor App")
st.write("Enter the details of the house to estimate its sale price:")

with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("""
    - **Model**: Random Forest Regressor  
    - **Features Used**:  
      `GrLivArea`, `GarageCars`, `OverallQual`,  
      `TotalBsmtSF`, `FullBath`, `YearBuilt`  
    - **R² Score**: `0.89`  
    - **RMSE**: ~$29,000  
    """)
    st.markdown("---")
    st.write("Built by Avanish")
    # st.write("Source Code: [GitHub](https://github.com/Avanish-Gupta/house-price-predictor)")

with st.form("prediction form"):
    st.subheader("Enter House Details")
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, step=50, value=1500)
    garage_cars = st.selectbox("Garage Capacity (number of cars)", options=[0, 1, 2, 3, 4], index=2)
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, step=50, value=800)
    full_bath = st.selectbox("Number of Full Bathrooms", options=[0, 1, 2, 3, 4], index=1)
    year_built = st.number_input("Year Built", min_value=1871, max_value=2010, step=1, value=2005)

    submit = st.form_submit_button("Predict Sale Price")

if submit:
    input_data = [[gr_liv_area, garage_cars, overall_qual, total_bsmt_sf, full_bath, year_built]]
    prediction = model.predict(input_data)[0]
    
    st.success(f"🏠 Estimated Sale Price: ${prediction:,.2f}")

    