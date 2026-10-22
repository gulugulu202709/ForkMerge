import argparse,json
import torch
from common import config
from models import make,scope_loss
from train import seed_all,state_hash
from physics import verify

def run(device):
    seed_all(731);x=torch.randn(8,12,3072,device=device);y=torch.randn(2,4,9,device=device)
    equal=torch.zeros(2,6,9,dtype=torch.bool,device=device);equal[:,0]=True
    hashes=[];params={}
    for arm in config()['arms']:
        seed_all(731);dim=4 if arm=='decoded_pair' else 3072;m=make(arm,dim).to(device)
        if arm!='decoded_pair':hashes.append(state_hash(m.project))
        p=m(x[...,:dim]);assert p.shape==(8,9) and torch.isfinite(p).all()
        loss=(p.reshape(2,4,9)-y).square().mean()+scope_loss(p.reshape(2,4,9),y,equal)
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        params[arm]=sum(p.numel() for p in m.parameters())
        if arm=='canonical_scope':
            m.eval();xx=x.clone();xx[:,1:6]+=torch.randn_like(xx[:,1:6])*5
            with torch.no_grad():gap=float((m(x,forced_gate=0)-m(xx,forced_gate=0)).abs().max())
            assert gap==0,'Forbidden old-history bypass'
    assert len(set(hashes))==1
    result={'device':device,'device_name':torch.cuda.get_device_name() if device=='cuda' else 'cpu',
      'all_heads_loss_backward':True,'shared_initial_projection':True,'gate0_history_invariance':True,
      'trainable_parameters':params,'physics':verify()}
    print('DIRECT_WITNESS_PASS',json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');run(p.parse_args().device)
