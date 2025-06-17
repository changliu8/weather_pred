import numpy as np
import pandas as pd
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df


def normalize(data,train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

def split_dataset(X,Y,split=0.8):
    train_index = int(X.shape[0]*split)
    validate_index = int((X.shape[0] - train_index)*split)

    X_train = X[:train_index]
    Y_train = Y[:train_index]

    X_val = X[train_index:train_index+validate_index,:]
    Y_val = Y[train_index:train_index+validate_index]

    X_test = X[train_index+validate_index:,:]
    Y_test = Y[train_index+validate_index:,:]







if __name__ == "__main__":
    current_directory = os.getcwd()
    filepath = os.path.join(os.getcwd(),"jena_climate_2009_2016.csv")
    df = load_data(filepath)

    df['Date Time'] = pd.to_datetime(df['Date Time'], format="%d.%m.%Y %H:%M:%S", dayfirst=True)
    df.set_index("Date Time",inplace=True)

    df_hourly = df.resample('h').mean().ffill()

    df_new = df_hourly.copy()

    df_new.info()

    #df_new = df_new.drop(['wv (m/s)', 'max. wv (m/s)', 'wd (deg)'],axis = 1)

    df_new = df_new.drop(['p (mbar)','wv (m/s)', 'max. wv (m/s)', 'wd (deg)'],axis = 1)

    X = df_new.to_numpy()
    Y = df_new['T (degC)'].to_numpy()

    split_dataset(X,Y)







    
