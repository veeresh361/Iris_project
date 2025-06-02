import torch
import torch.nn as nn
import torch.nn.functional as F



class VeereshNetwork(nn.Module):

    def __init__(self,input_dim,hidden_layers,output_dim):
        super(VeereshNetwork,self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim,hidden_layers[0]))
        for i in range(1,len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(hidden_layers[i-1],hidden_layers[i]))
        self.output_layer= nn.Linear(hidden_layers[-1],output_dim)

    def forward(self,x):
        for layer in self.hidden_layers:
            x=layer(x)
            x=F.relu(x)
        x=self.output_layer(x)
        x = torch.sigmoid(x) 
        return x

        
