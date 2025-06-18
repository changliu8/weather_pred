import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df


def normalize(data,train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

def pre_process_dataset(X,Y,split=0.7):
    # split the dataset

    train_index = int(X.shape[0]*split)
    validate_index = int((X.shape[0] - train_index)*split)

    X_train = X[:train_index]
    Y_train = Y[:train_index]

    X_val = X[train_index:train_index+validate_index,:]
    Y_val = Y[train_index:train_index+validate_index]

    X_test = X[train_index+validate_index:,:]
    Y_test = Y[train_index+validate_index:]

    #scale the dataset
    X_train_mean = np.mean(X_train,axis=0)
    Y_train_mean = np.mean(Y_train)

    X_train_std = np.std(X_train,axis=0)
    Y_train_std = np.std(Y_train)

    X_train_norm = (X_train - X_train_mean) / X_train_std
    X_val_norm = (X_val - X_train_mean) / X_train_std
    X_test_norm = (X_test - X_train_mean) / X_train_std

    Y_train_norm = (Y_train - Y_train_mean) / Y_train_std
    Y_val_norm = (Y_val - Y_train_mean) / Y_train_std
    Y_test_norm = (Y_test - Y_train_mean) / Y_train_std

    x_train = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train = torch.tensor(Y_train_norm,dtype=torch.float32)

    x_val = torch.tensor(X_val_norm,dtype=torch.float32)
    y_val = torch.tensor(Y_val_norm,dtype=torch.float32)

    x_test = torch.tensor(X_test_norm,dtype=torch.float32)
    y_test = torch.tensor(Y_test_norm,dtype=torch.float32)


    return x_train,y_train,x_val,y_val,x_test,y_test

def data_seq(x_train,y_train,x_val,y_val,x_test,y_test,seq_l=24):
    # the first contains data from 0-23
    # the second contains data from 1-24
    # the third contains data from 2-25

    x_train_seq = torch.stack([x_train[i:i+seq_l,:]for i in range(x_train.shape[0]-seq_l)])
    x_val_seq = []
    for i in range(x_val.shape[0]-seq_l):
        x_val_seq.append(x_val[i:i + seq_l, :])
    x_val_seq = torch.stack(x_val_seq)
    x_test_seq = torch.stack([x_test[i:i+seq_l,:] for i in range(x_test.shape[0]-seq_l)])
    y_train_seq = torch.stack([y_train[i+seq_l] for i in range(y_train.shape[0]-seq_l)])
    y_val_seq = torch.stack([y_val[i+seq_l] for i in range(y_val.shape[0]-seq_l)])
    y_test_seq = torch.stack([y_test[i+seq_l] for i in range(y_test.shape[0]-seq_l)])
    

    return x_train_seq,x_val_seq,x_test_seq,y_train_seq,y_val_seq,y_test_seq

def create_tensor(x_train_seq,y_train_seq,x_val_seq,y_val_seq,x_test_seq,y_test_seq):
    train_dataset = TensorDataset(x_train_seq,y_train_seq)
    val_dataset = TensorDataset(x_val_seq,y_val_seq)
    test_dataset = TensorDataset(x_test_seq,y_test_seq)

    return train_dataset,val_dataset,test_dataset

def create_dataloader(train_dataset,val_dataset,test_dataset,batch_size=16):
    train_dataloader = DataLoader(train_dataset,batch_size = batch_size, drop_last=False,shuffle=False)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,drop_last=False,shuffle=False)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,drop_last=False,shuffle=False)
    return train_dataloader,val_dataloader,test_dataloader

def check_size(dataloader):
    for batch in dataloader:
        x,y = batch
        print(x.shape,y.shape)
        break

if __name__ == "__main__":
    current_directory = os.getcwd()
    filepath = os.path.join(os.getcwd(),"jena_climate_2009_2016.csv")
    df = load_data(filepath)

    df['Date Time'] = pd.to_datetime(df['Date Time'], format="%d.%m.%Y %H:%M:%S", dayfirst=True)
    df.set_index("Date Time",inplace=True)

    df_hourly = df.resample('h').mean().ffill()

    df_new = df_hourly.copy()

    #df_new = df_new.drop(['wv (m/s)', 'max. wv (m/s)', 'wd (deg)'],axis = 1)

    df_new = df_new.drop(['p (mbar)','wv (m/s)', 'max. wv (m/s)', 'wd (deg)'],axis = 1)
    #X = df_new.drop(['T (degC)'],axis=1).to_numpy()
    X = df_new.to_numpy()
    Y = df_new['T (degC)'].to_numpy()

    x_train,y_train,x_val,y_val,x_test,y_test = pre_process_dataset(X,Y)
    x_train_seq,x_val_seq,x_test_seq,y_train_seq,y_val_seq,y_test_seq = data_seq(x_train,y_train,x_val,y_val,x_test,y_test)
    train_dataset,val_dataset,test_dataset = create_tensor(x_train_seq,y_train_seq,x_val_seq,y_val_seq,x_test_seq,y_test_seq)
    train_dataloader,val_dataloader,test_dataloader = create_dataloader(train_dataset,val_dataset,test_dataset)





    
