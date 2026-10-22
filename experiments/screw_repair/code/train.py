import argparse,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from common import ROOT,config,dump,sha,PAIRS
from models import make,scope_loss

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)

def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def load(draw,split,device='cpu',confirmation=False):
    root=ROOT if confirmation else Path(config()['source_root'])
    p=root/'features/siglip2'/f'draw{draw}'/(split+'.npz')
    with np.load(p) as d:x=d['x'];y=d['targets']
    return torch.as_tensor(x,device=device),torch.as_tensor(y,device=device),p

def predict(model,x):
    model.eval()
    with torch.inference_mode():return torch.cat([model(v.flatten(0,1)).reshape(-1,4,9) for v in x.split(32)])

def fit(draw,arm,seed,lr,device='cuda'):
    cfg=config();assert arm in ('tf_scope','canonical_scope')
    out=ROOT/'results'/f'draw{draw}'/f'{arm}_s{seed}_lr{lr:g}';out.mkdir(parents=True,exist_ok=False)
    seed_all(seed);x,y,tp=load(draw,'train',device);vx,vy,vp=load(draw,'validation',device)
    fm=x.mean((0,1,2));fs=x.std((0,1,2),unbiased=False).clamp_min(.01)
    x=(x-fm)/fs;vx=(vx-fm)/fs;mean=y.mean((0,1));std=y.std((0,1),unbiased=False).clamp_min(.01)
    yy=(y-mean)/std;vyy=(vy-mean)/std
    equal=torch.stack([(y[:,a]-y[:,b]).abs()<1e-6 for a,b in PAIRS],1)
    model=make(arm,x.shape[-1],cfg['hidden']).to(device);initial=state_hash(model);project=state_hash(model.project)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    history=[];best=float('inf');best_epoch=None;steps=0;start=time.time()
    for epoch in range(1,cfg['epochs']+1):
        model.train();losses=[]
        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed*100000+epoch))
        for ix in order.split(cfg['batch_families']):
            ii=ix.to(device);opt.zero_grad(set_to_none=True)
            p=model(x[ii].flatten(0,1)).reshape(-1,4,9);mse=(p-yy[ii]).square().mean()
            loss=mse+cfg['pair_lambda']*scope_loss(p,yy[ii],equal[ii])
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();steps+=1;losses.append(float(mse.detach()))
        vmse=float((predict(model,vx)-vyy).square().mean())
        history.append({'epoch':epoch,'steps':steps,'train_MSE':float(np.mean(losses)),'validation_MSE':vmse,
          'sample_order_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest(),'seconds':time.time()-start})
        if vmse<best:
            best=vmse;best_epoch=epoch
            torch.save({'model':model.state_dict(),'feature_mean':fm.cpu(),'feature_std':fs.cpu(),'target_mean':mean.cpu(),
              'target_std':std.cpu(),'dim':x.shape[-1],'head':'canonical' if arm=='canonical_scope' else 'transformer',
              'selected_epoch':epoch,'validation_MSE':vmse,'arm':arm,'draw':draw,'seed':seed,'lr':lr,'budget':cfg['epochs']},out/'best.pt')
        if epoch%100==0:print('TRAIN',draw,arm,seed,lr,epoch,vmse,flush=True)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'steps':steps},out/'last.pt')
    dump(out/'history.json',history)
    dump(out/'FIT.json',{'arm':arm,'draw':draw,'seed':seed,'lr':lr,'epochs':epoch,'steps':steps,
      'initial_state_sha256':initial,'initial_projection_sha256':project,'parameters':sum(p.numel() for p in model.parameters()),
      'best_epoch':best_epoch,'best_validation_MSE':best,'seconds':time.time()-start,
      'inputs':{str(p):sha(p) for p in [tp,vp]},'checkpoint_sha256':sha(out/'best.pt'),
      'protocol_sha256':sha(ROOT/'configs/protocol.json'),'code_hashes':__import__('common').code_hashes()})
    print('FIT_COMPLETE',draw,arm,seed,lr,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draw',type=int,required=True);p.add_argument('--arm',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();fit(a.draw,a.arm,a.seed,a.lr,a.device)
