import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from network import NeuralNetwork
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset,DataLoader

from sklearn.preprocessing import MinMaxScaler

def pre_processing(df):
    #normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    #sequence length and features
    sequence_length = 10

    #create sequences and corresponding labels
    sequences = []
    labels = []
    for i in range(len(scaled_data)-sequence_length):
        seq = scaled_data[i:i+sequence_length]
        label = scaled_data[i+sequence_length][6]
        sequences.append(seq)
        labels.append(label)
    #convert to numpy array
    sequences = np.array(sequences)
    labels = np.array(labels)

    #split into train,val and test sets
    total_size = len(sequences)
    train_size = int(0.65*total_size)
    val_size = int(0.15*total_size)

    train_x,val_x,test_x = sequences[:train_size],sequences[train_size:train_size+val_size],sequences[train_size+val_size:]
    train_y,val_y,test_y = labels[:train_size],labels[train_size:train_size+val_size],labels[train_size+val_size:]

    print(train_x.shape,val_x.shape,test_x.shape)

    print("train_x shape is : {} , train_y shape is : {}".format(train_x.shape,train_y.shape))
    print("train_x shape is : {} , train_y shape is : {}".format(val_x.shape,val_y.shape))
    print("test_x shape is : {} , test_y shape is : {}".format(test_x.shape,test_y.shape))

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    train_dataset = TensorDataset(train_x,train_y)
    val_dataset = TensorDataset(val_x,val_y)
    test_dataset = TensorDataset(test_x,test_y)
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

    return train_loader,val_loader,test_loader


    

def load_data(filepath,drop=False):
    df = pd.read_csv(filepath)
    if drop:
        df = df.dropna()
    return df


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(),'testset.csv')
    df = load_data(file_path)
    df.columns = df.columns.str.lstrip()
    df.columns = df.columns.str.lstrip()
    #set index to the time instead of sequences(e.g. 0,1,2,3,...)
    df.index = pd.to_datetime(df.datetime_utc)
    
    required_columns = ['_dewptm','_fog','_hail','_hum','_rain','_snow','_tempm','_thunder','_tornado']
    df = df[required_columns]

    df_final = df.fillna(method = 'ffill')

    train_loader,val_loader,test_loader = pre_processing(df_final)
    '''
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y,dtype=torch.float32)

    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y,dtype=torch.float32)
    '''

    model = NeuralNetwork(9,128,1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


    #indice = list(range(len(test_x)))
    #indice = list(range(300))

    num_epochs = 100
    h0,c0 = None,None

    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x,batch_y in train_loader:
            optimizer.zero_grad()
            output,h0,c0 = model(batch_x)
            loss = criterion(output.view(-1),batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)

        #validating step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                output, _, _ = model(val_x)
                loss = criterion(output.view(-1),val_y)
                val_loss+=loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict, "best_model.pt")
        else:
            counter +=1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break


    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output, _, _ = model(batch_x)
            predictions.append(output.view(-1))
            true_labels.append(batch_y)

    predictions = torch.cat(predictions).numpy()
    true_labels = torch.cat(true_labels).numpy()

    plt.plot(true_labels,linestyle="none",label="True",marker='o')
    plt.plot(predictions, linestyle='none',label="Predicted", marker='x')
    plt.title('LSTM Model Predictions vs. Original Data')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()








