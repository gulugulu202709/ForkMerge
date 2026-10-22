import torch
from torch import nn

class Model(nn.Module):
    def __init__(self,dim,kind,hidden=128):
        super().__init__();self.kind=kind
        self.project=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        if kind=='gru': self.sequence=nn.GRU(hidden,hidden,batch_first=True)
        else:
            layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=0.,batch_first=True,norm_first=True)
            self.sequence=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.pos=nn.Parameter(torch.randn(12,hidden)*.01)
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,9))
    def forward(self,x):
        z=self.project(x)
        z=self.sequence(z)[1][-1] if self.kind=='gru' else self.sequence(z+self.pos[None])[:,-1]
        return self.head(z)
