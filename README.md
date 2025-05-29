# 🏡 House Price Prediction App

A machine learning web application that predicts house sale prices using key engineered features from the Ames Housing Dataset. The model is built with a Random Forest Regressor and deployed using Streamlit Cloud.

---

## 🔧 Tech Stack

- **Python**: Data manipulation and machine learning
- **Pandas, NumPy, Seaborn**: Data preprocessing and visualization
- **scikit-learn**: Model building and evaluation
- **Streamlit**: Web application framework
- **Joblib**: Model serialization
- **Git/GitHub**: Version control and deployment integration

---

## 🎯 Features

- Real-time price prediction using:
  - `GrLivArea` (Above Ground Living Area)
  - `GarageCars` (Garage Capacity)
  - `OverallQual` (Overall Quality Rating)
  - `TotalBsmtSF` (Basement Area)
  - `FullBath` (Number of Full Bathrooms)
  - `YearBuilt` (Year Constructed)

- Model Performance:
  - **R² Score**: 0.89
  - **RMSE**: ~$29,000

- Interactive UI for user input

---

## 📦 Installation (Local)

```bash
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
cd house-price-predictor
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

This project is live on Streamlit Cloud
https://house-price-predictor-ml-model-jtimfpkk5sagxkmc43bado.streamlit.app/

👨‍💻 Author
Avanish Suthakaran
Bachelor of Software Engineering
Western University
