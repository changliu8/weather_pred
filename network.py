import torch

from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(NeuralNetwork,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = 3
        self.lstm1 = nn.LSTM(input_dim,hidden_dim,num_layers=self.layer_dim,batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        #self.lstm2 = nn.LSTM(hidden_dim,hidden_dim)
        #self.lstm3 = nn.LSTM(hidden_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)

    
    def forward(self,x,h0=None,c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn,cn) = self.lstm1(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out,hn,cn


