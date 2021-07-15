import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.nn.Module:It is a base class used to develop all neural network models.
#torch.nn.Linear()	This module applies the linear transformation of the input data.
#torch.nn.ReLU()	It applies ReLU function and takes the max between 0 and element :ReLU(x)=max(0,x)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers #stocker le nbre des layers
        self.hidden_size = hidden_size #stocker hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True)
        #x -> (batch_size, sequence_length, input_size)

        self.fc = nn.Linear(hidden_size, num_classes) # create a linear layer

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)#the initial tensor for the cell state (le représentant de la mémoire à long terme)

        out, _ = self.lstm(x, (h0, c0)) # call our LSTM model
        #the shape of output is (batch_size, sequence_length, hidden_size)

        out = out[:, -1, :]
        out = self.fc(out)
        # no activation and no softmax at the end
        return out