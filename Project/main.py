from modules.data_loading import DataLoader
from modules.data_preprocessing import DataCleaner
from modules.data_statistics import DataStatistics
from modules.univariate_analysis import UnivariateAnalysis
from modules.univariate_plot import UnivariatePlots
from modules.bivariate_analysis import BivariateAnalysis
from modules.bivariate_plot import BivariatePlots
from modules.data_splitter import DataSplitter
from modules.model_trainer import ModelTrainer
from modules.model_evaluator import ModelEvaluator
from modules.model_predictor import Predictor
from modules.pickles import PickleHandler

"""Initialize DataLoader Object with CSV File"""
loader = DataLoader("diabetes.csv")
print(loader.load_data())
print(loader.show_head())
print(loader.show_tail())
print(loader.show_sample())
print(loader.show_description())
print(loader.show_info())
 

loader = DataLoader("diabetes.csv")
loader.load_data()
df = loader.df
cleaner = DataCleaner(df)
print(cleaner.check_nulls())
print(cleaner.replace_zeros())
print(cleaner.rename_columns())

# Get the cleaned and updated dataframe
cleaned_df = cleaner.get_clean_data()
print(cleaned_df)
# Assuming 'cleaned_df' is the dataframe after cleaning
stats = DataStatistics(cleaned_df)

print(stats.show_mean())
print(stats.show_median())
print(stats.show_mode())
print(stats.show_max())
print(stats.show_min())
print(stats.show_std())
print(stats.show_variance())
print(stats.show_count())


# Univariate
uni = UnivariateAnalysis(cleaned_df)
print(uni.unique_counts('Age'))
print(uni.value_counts('Glu'))
print(uni.column_summary('BMI'))
print(uni.check_skewness('BP'))
print(uni.check_kurtosis('Ins'))


# Bivariate
bi = BivariateAnalysis(cleaned_df)
print(bi.correlation_matrix())
print(bi.covariance_matrix())
print(bi.group_mean_by_target('Outcome'))
print(bi.group_count_by_target('Outcome'))
print(bi.crosstab_two_columns('Preg', 'Outcome'))



plotter = UnivariatePlots(cleaned_df)

# Histogram with custom color and bins
plotter.plot_histogram("Glu", bins=20, color="orange", edgecolor="gray", figsize=(10, 5))

# Boxplot horizontal
plotter.plot_boxplot("BMI", color="green", figsize=(8, 4), orient='h')

# KDE Plot with custom shade color
plotter.plot_kde("Age", color="navy", shade=True, figsize=(10, 5))

# Countplot for Outcome (categorical)
plotter.plot_countplot("Outcome", palette="pastel", figsize=(7, 4))

plotter = BivariatePlots(cleaned_df)

# Custom scatter plot
plotter.scatter_plot("Glu", "Ins", palette='viridis', figsize=(10, 6))


# 🔸 Correlation heatmap with different cmap
plotter.correlation_heatmap(cmap='magma', figsize=(12, 10))

# 🔸 Boxplot with custom size and palette
plotter.boxplot_by_category("BMI", palette="spring", figsize=(8, 5))


# Splitting the data
splitter = DataSplitter(cleaned_df, target_column='Outcome')
X_train, X_test, y_train, y_test = splitter.split_data(test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

# Step 2: Training the model
trainer = ModelTrainer()
trainer.train_model(X_train, y_train)


# Evaluating the model
evaluator = ModelEvaluator(trainer.model, X_test, y_test)
print(evaluator)


# Show predictions
predictions = evaluator.show_prediction()
print(predictions)

# Accuracy score
accuracy = evaluator.show_accuracy()
print(accuracy)

    
print(evaluator.show_classification_report())

# Confusion matrix
print(evaluator.show_confusion_matrix())

# 4. Predict on new/test data (Optional)

predictor = Predictor(trainer.model)

import pandas as pd
new_data = pd.DataFrame({
    'Preg': [2],
    'Glu': [130],
    'BP': [80],
    'Skin': [25],
    'Ins': [100],
    'BMI': [30.5],
    'DPF': [0.4],
    'Age': [35]
})

# Prediction (0 = No Diabetes, 1 = Diabetes)
result = predictor.predict(new_data)
print("Predicted Outcome:", "Diabetic" if result[0] == 1 else "Not Diabetic")


# 1. After training the model
pickle_handler = PickleHandler()
pickle_handler.save_model('diabetes_model.pkl')

# # Predict using the loaded model
# predictor = pickle_handler.load_model()

# Predictor(predictor)  

# # Example input for prediction
# new_input = pd.DataFrame({
#     'Preg': [4],
#     'Glu': [140],
#     'BP': [70],
#     'Skin': [20],
#     'Ins': [80],
#     'BMI': [33.0],
#     'DPF': [0.6],
#     'Age': [45]
# })

# result = predictor.predict(new_input)
# print("Prediction from Pickled Model:", "Diabetic" if result[0] == 1 else "Not Diabetic")
