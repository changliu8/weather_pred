import numpy as np
import pandas as pd
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("The data contains {} data in total.".format(len(df)))
    print(df.columns.tolist())






if __name__ == "__main__":
    current_directory = os.getcwd()
    filepath = os.path.join(os.getcwd(),"jena_climate_2009_2016.csv")
    load_data(filepath)
