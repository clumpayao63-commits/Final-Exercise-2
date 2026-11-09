# Wine Quality Prediction System

## Problem
A small wine-production business needs a preliminary way to screen red-wine samples using physicochemical measurements.

## Dataset
UCI Wine Quality Dataset — Red Wine CSV:
https://archive.ics.uci.edu/dataset/186/wine+quality

## Prediction rule
- Quality score >= 6: GOOD QUALITY
- Quality score < 6: NEEDS IMPROVEMENT

The original `quality` column is used only to create the target. It is not an input feature.

## Technologies
Python, Pandas, Scikit-learn, Random Forest, Joblib, Streamlit, and GitHub.

## Installation
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install packages:

   pip install -r requirements.txt

## Train the model

   python train_model.py

This downloads `wine_quality.csv`, trains and evaluates the model, and creates `wine_model.pkl`.

## Run the application

   streamlit run app.py

Open the local address shown in the terminal, normally http://localhost:8501.

## Limitations
This classroom prototype is trained on a particular red-wine dataset. It may not generalize to other wine types, producers, regions, or testing conditions. Predictions are preliminary screening results and are not a substitute for laboratory testing or professional quality-control decisions.
