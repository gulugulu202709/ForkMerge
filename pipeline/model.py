"""Shared ordinary updater and two loss weightings. No native model import."""
import torch
from torch import nn
from .config import ACTIONS, NEW_COEFFICIENT, PAIR_COEFFICIENT, TOKENS, WIDTH

class OrdinaryUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp=nn.Linear(WIDTH,256)
        self.action=nn.Embedding(len(ACTIONS),256)
        self.pos=nn.Parameter(torch.randn(1,2*TOKENS,256)*.01)
        layer=nn.TransformerEncoderLayer(256,8,512,dropout=0.,batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.new=nn.Linear(256,WIDTH);self.gate=nn.Linear(256,WIDTH)
        self.out=nn.Linear(WIDTH,WIDTH);self.norm=nn.LayerNorm(WIDTH)
        self.scale=nn.Parameter(torch.tensor(.1))
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,features,actions,trace=False):
        if features.ndim!=4 or tuple(features.shape[1:])!=(4,TOKENS,WIDTH):raise ValueError('history feature shape')
        if tuple(actions.shape)!=(len(features),4):raise ValueError('action shape')
        m=features[:,0];states=[m]
        for k in range(1,4):
            a=self.action(actions[:,k])[:,None]
            x=torch.cat([self.inp(m),self.inp(features[:,k])+a],1)+self.pos
            h=self.tr(x)[:,:TOKENS]
            gate=torch.sigmoid(self.gate(h))
            m=gate*m+(1-gate)*self.new(h);states.append(m)
        final=(self.norm(m)+self.out(m))*self.scale
        return (final,states) if trace else final

def paired_objective(predictions,targets,masks,arm):
    """3 actual future flow targets,3readout masks. Same graph for both arms."""
    if arm not in ('paired','paired_worst_witness'):raise ValueError('unknown arm')
    if predictions.shape!=targets.shape or predictions.ndim!=5 or predictions.shape[0]!=3:raise ValueError('triplet flow shape')
    if tuple(masks.shape)!=(3,*predictions.shape[2:]) or torch.any(masks<0) or torch.any(masks.sum((1,2,3))<=0):raise ValueError('witness masks')
    errors=predictions-targets
    flow=errors.square().mean()
    pairs=[(0,1),(0,2),(1,2)]
    pair=torch.stack([(errors[i]-errors[j]).square().mean() for i,j in pairs]).mean()
    relations=[]
    for mask in masks:
        denom=mask.sum()*predictions.shape[1]
        terms=[]
        for i,j in pairs:
            true_distance=((targets[i]-targets[j]).square()*mask).sum()/denom
            pred_distance=((predictions[i]-predictions[j]).square()*mask).sum()/denom
            terms.append(((pred_distance+1e-8).sqrt()-(true_distance+1e-8).sqrt()).square()/(true_distance+.01))
        relations.append(torch.stack(terms).mean())
    worst=torch.stack(relations).max()
    loss=flow+PAIR_COEFFICIENT*pair+(NEW_COEFFICIENT if arm=='paired_worst_witness' else 0.)*worst
    return loss,{'proper_flow':float(flow.detach()),'paired':float(pair.detach()),'worst_relation':float(worst.detach())}

def _overlap_weights(bounds,n,device):
    lo,hi=map(float,bounds)
    if not 0<=lo<hi<=1:raise ValueError('normalized witness bounds')
    left=torch.arange(n,device=device,dtype=torch.float32)/n
    return (torch.minimum(left+1/n,torch.tensor(hi,device=device))-torch.maximum(left,torch.tensor(lo,device=device))).clamp_min(0)*n

def witness_masks(clips,shape,device='cpu'):
    """Approximate source readout support on VAE latent grid; not semantic masks."""
    _,t,h,w=shape;out=[]
    for side in ('left','right'):
        per_clip=[]
        for clip in clips:
            x0,y0,x1,y1=clip['rois'][side]
            tw=_overlap_weights(clip['window_fractions'][side],t,device)
            yw=_overlap_weights((y0,y1),h,device);xw=_overlap_weights((x0,x1),w,device)
            per_clip.append(tw[:,None,None]*yw[None,:,None]*xw[None,None,:])
        out.append(torch.stack(per_clip).mean(0))
    out.append(torch.ones(t,h,w,device=device))
    return torch.stack(out)

def trace_statistics(final,states):
    """Only descriptive statistics of supplied actual tensors, no semantic labels."""
    return {'shape':list(final.shape),'stage_norms':[s.detach().float().norm(dim=-1).cpu().tolist() for s in states],
            'stage_deltas':[(b-a).detach().float().norm(dim=-1).cpu().tolist() for a,b in zip(states,states[1:])],
            'final_norms':final.detach().float().norm(dim=-1).cpu().tolist(),
            'meaning':'Tensor changes only; not evidence of semantic erase/protect or causal use.'}
