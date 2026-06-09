# Sales Prediction using XGBoost

## Overview

This project focuses on predicting weekly sales using the Walmart Sales Dataset from Kaggle. The goal is to analyze historical sales data, discover important business insights, and build a machine learning model that can forecast future sales.

## Dataset

* Dataset: Walmart Sales Dataset
* Source: Kaggle
* Target Variable: Weekly Sales

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Analyzed sales trends over time
* Compared holiday and non-holiday sales
* Studied the impact of fuel prices and temperature on sales
* Identified seasonal patterns

### 2. Data Preprocessing

* Handled missing values
* Converted date columns into useful features
* Prepared data for model training

### 3. Feature Engineering

* Created lag features using previous sales values
* Extracted date-related features such as month and year
* Generated features to improve model performance

### 4. Model Training

* Used XGBoost Regressor
* Trained the model on historical sales data
* Evaluated performance using regression metrics

## Model Performance

* Optimized MAE: $38,593.04

## Key Insights

* The previous year recorded the highest sales.
* Sales increased significantly during November and December.
* Holiday weeks generated approximately 8.6% higher sales than normal weeks.
* Temperature and fuel prices showed little impact on weekly sales.

## Key Learning

This project helped me understand how Time Series Forecasting differs from traditional Machine Learning problems.

* Historical data plays a crucial role in prediction.
* Lag features are important for capturing past sales patterns.
* Data should be split based on time rather than randomly.
* Proper time-based splitting helps prevent data leakage.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

## Future Improvements

* Hyperparameter tuning
* Advanced time series models
* Additional feature engineering
* Model deployment using Streamlit

## Author

S. NAGAVAMSHI
AIML Student | Machine Learning & Data Analytics Enthusiast
