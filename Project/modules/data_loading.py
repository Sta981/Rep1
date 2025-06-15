
# Numerical and data manipulation 
import numpy as np
import pandas as pd


class DataLoader:

    def __init__(self, file_path):
        """
        Initializes the DataLoader with file path
        Loads data into self.df
        """
        self.file_path = file_path
        self.df = None 
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            return "Data loaded successfully"
        except Exception as e:
            return f"Error loading data : {e} "
        
    def show_head(self):
        print("\nHead of the dataset:")
        return self.df.head()

    def show_tail(self):
        print("\nTail of the dataset:")
        return self.df.tail()
    
    def show_shape(self):
        print("\nRows and columns in the dataset are:")
        return self.df.shape
    def show_sample(self):
        print("\nRandom Sample of the Dataset :")
        return self.df.sample()

    def show_description(self):
        print("\nStatistical summary:")
        return self.df.describe()
        
    def show_info(self):
        print("/nDataset information :")
        return self.df.info()

