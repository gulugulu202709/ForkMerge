"""Continuous-response adaptation of the paper canonical+witness interface."""
import math
import torch
from torch import nn
from baseline_model import Model as Baseline

class CanonicalWitness(nn.Module):
    def __init__(self,dim,hidden=128):
        super().__init__()
        # First module matches the baseline input projection and seed exactly.
        self.project=nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.prefix=nn.GRU(hidden,hidden,batch_first=True)
        self.context=nn.GRU(hidden,hidden,batch_first=True)
        self.query_features=nn.Linear(hidden,3*64)
        self.memory_keys=nn.Linear(hidden,64)
        self.memory_values=nn.Sequential(nn.Conv1d(hidden,128,3,padding=1),nn.GELU(),nn.Conv1d(128,64,1))
        self.memory_queries=nn.Linear(hidden,3*64)
        self.memory_positions=nn.Parameter(torch.randn(6,64)*.02)
        self.memory_gate=nn.Linear(hidden+128,64)
        self.readout=nn.Linear(64,3)
        nn.init.zeros_(self.memory_values[-1].weight);nn.init.zeros_(self.memory_values[-1].bias)
        nn.init.zeros_(self.memory_gate.weight);nn.init.zeros_(self.memory_gate.bias)

    def forward(self,x,forced_gate=None):
        z=self.project(x);history=z[:,:6]
        pre=self.prefix(history)[1][-1]
        # Calibration carries invariant material stiffness to every query.
        ctx=self.context(torch.cat([z[:,:1],z[:,6:]],1))[1][-1]
        query=self.memory_queries(ctx).reshape(-1,3,64)
        key=self.memory_keys(history)+self.memory_positions[None]
        attention=torch.einsum('bqd,btd->bqt',query,key).div(math.sqrt(64)).softmax(-1)
        value=self.memory_values(history.transpose(1,2)).transpose(1,2)
        read=torch.einsum('bqt,btd->bqd',attention,value)
        witness=self.query_features(pre).reshape(-1,3,64)+read
        canonical=self.query_features(ctx).reshape(-1,3,64)
        gate=self.memory_gate(torch.cat([ctx[:,None].expand(-1,3,-1),query,witness],-1)).sigmoid()
        if forced_gate is not None:gate=torch.full_like(gate,float(forced_gate))
        functional=canonical+gate*witness
        self.last_gate=gate;self.last_canonical=canonical;self.last_witness=witness
        self.last_functional=functional
        return self.readout(torch.nn.functional.gelu(functional)).flatten(1)

def make(arm,dim,hidden=128):
    if arm=='canonical_scope':return CanonicalWitness(dim,hidden)
    return Baseline(dim,'gru' if arm=='decoded_pair' else 'transformer',hidden)

def scope_loss(pred,targets,equal):
    from common import PAIRS
    delta=torch.stack([pred[:,a]-pred[:,b] for a,b in PAIRS],1)
    td=torch.stack([targets[:,a]-targets[:,b] for a,b in PAIRS],1)
    erase=(delta.square()*equal).sum()/equal.sum().clamp_min(1)
    keep=~equal
    protect=((delta-td).square()*keep).sum()/keep.sum().clamp_min(1)
    return erase+protect
