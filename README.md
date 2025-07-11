# House-Price-Prediction

This repository contains a simple machine learning project that predicts house prices based on various features such as area, number of bedrooms, bathrooms, and parking. The model is built using Python and trained using a supervised learning approach.

## 📌 Overview

This project demonstrates how machine learning can be used to make predictions on real-world data. A linear regression model is trained using scikit-learn to predict the price of a house given its features.

## 🔧 Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (optional - for data visualization)

## 📁 Dataset

The dataset used is a simple CSV file (`HousePricePrediction.xlsx`) that contains the following columns:

- `area` – total area in square feet
- `bedrooms` – number of bedrooms
- `bathrooms` – number of bathrooms
- `stories` – number of floors
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus` – categorical and binary features
- `price` – target variable (house price)

## 🧠 Model Used

- **Linear Regression** from `sklearn.linear_model`

## 🔍 How It Works

1. Load and preprocess the dataset.
2. Encode categorical variables.
3. Split data into training and testing sets.
4. Train a **Linear Regression** model.
5. Make predictions and evaluate accuracy.



