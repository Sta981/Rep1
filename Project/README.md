# Diabetes Prediction Project

This project analyzes the Pima Indians Diabetes dataset to predict whether a person has diabetes based on diagnostic measurements. The project was created for learning purposes using Python and common data science tools.

## Dataset

The dataset is taken from Kaggle:
https://www.kaggle.com/datasets/mathchi/diabetes-data-set

It contains medical and personal data for several patients, including:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target: 1 = diabetic, 0 = non-diabetic)

## Objectives

- Perform basic data exploration and cleaning
- Visualize important patterns and relationships
- Train machine learning models to predict diabetes
- Build an interactive Streamlit dashboard for user-friendly prediction

## Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

## Project Structure

- **Project 2.0.ipynb**: Jupyter Notebook with full data analysis and model training
- **diabetes.csv**: Dataset file used in the project
- **diabetes_dashboard.py**: Streamlit app for interactive prediction and visualizations

## Key Features

- Exploratory data analysis with histograms, boxplots, heatmaps, and more
- Preprocessing using standard scaling
- Model training using Random Forest Classifier
- Live prediction using slider inputs in Streamlit
- Multiple visualizations controlled through checkboxes in the dashboard

## How to Run the Streamlit App

1. Make sure `diabetes.csv` is in the same directory.
2. Install required libraries:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
