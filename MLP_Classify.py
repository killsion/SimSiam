import torch
import torch.nn as nn



class MLP(nn.Module):

    def __init__(self,input_dim,out_put):

        super(MLP,self).__init__()
        hidden=int(0.5*input_dim)
        self.linar_1=nn.Linear(input_dim,hidden,bias=True)
        self.linar_2=nn.Linear(hidden,out_put,bias=True)
        self.relu=nn.ReLU()


    def forward(self,x):
        x=self.linar_1(x)
        x=self.relu(x)
        x=self.linar_2(x)


        return x


