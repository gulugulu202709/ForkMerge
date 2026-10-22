import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,config,dump,sha
from models import make
from baseline_model import Model as Baseline
from train import seed_all,load,predict

def freeze():
    cfg=config();items=json.loads((ROOT/'evidence/REUSED_BASELINES.json').read_text())['selected_checkpoints']
    for d in cfg['draws']:
      for a in ('tf_scope','canonical_scope'):
       for s in cfg['seeds']:
        opts=[]
        for lr in cfg['learning_rates']:
            path=ROOT/'results'/f'draw{d}'/f'{a}_s{s}_lr{lr:g}';info=json.loads((path/'FIT.json').read_text())
            assert info['epochs']==cfg['epochs'] and info['steps']==cfg['updates_per_fit']
            assert info['protocol_sha256']==sha(ROOT/'configs/protocol.json')
            assert sha(path/'best.pt')==info['checkpoint_sha256'];opts.append((path,info))
        path,info=min(opts,key=lambda v:(v[1]['best_validation_MSE'],v[1]['lr']))
        items.append({'draw':d,'arm':a,'seed':s,'budget':cfg['epochs'],'path':str(path/'best.pt'),
          'sha256':sha(path/'best.pt'),'selected_lr':info['lr'],'selected_epoch':info['best_epoch'],
          'origin':'new_matched_fit','encoder':'siglip2'})
    assert len(items)==30
    for v in items:assert sha(v['path'])==v['sha256']
    dump(ROOT/'evidence/FIT_FREEZE.json',{'selected_checkpoints':items,'all_new_fits_complete':True,
       'all_confirmation_predictions_after_freeze':True,'protocol_sha256':sha(ROOT/'configs/protocol.json')})
    return items

def evaluate(device='cuda'):
    cfg=config();items=json.loads((ROOT/'evidence/FIT_FREEZE.json').read_text())['selected_checkpoints'];rows=[]
    for item in items:
        seed_all(item['seed']);assert sha(item['path'])==item['sha256']
        ck=torch.load(item['path'],map_location=device,weights_only=False)
        m=Baseline(ck['dim'],'gru',cfg['hidden']) if item['arm']=='decoded_pair' else make(item['arm'],ck['dim'],cfg['hidden'])
        m=m.to(device);m.load_state_dict(ck['model'])
        for split in cfg['test_splits']:
            source=ROOT/'features'/item['encoder']/f'draw{item["draw"]}'/(split+'.npz')
            with np.load(source) as d:x=torch.as_tensor(d['x'],device=device);y=d['targets'];ids=d['family_ids']
            x=(x-ck['feature_mean'].to(device))/ck['feature_std'].to(device)
            p=(predict(m,x)*ck['target_std'].to(device)+ck['target_mean'].to(device)).cpu().numpy()
            dest=ROOT/'predictions'/f'draw{item["draw"]}'/f'{item["arm"]}_s{item["seed"]}_{split}.npz'
            dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,predictions=p,targets=y,family_ids=ids)
            rows.append({'path':str(dest.relative_to(ROOT)),'sha256':sha(dest),'feature_sha256':sha(source),'checkpoint_sha256':item['sha256']})
        print('EVALUATED_DIRECT',item['draw'],item['arm'],item['seed'],flush=True)
    dump(ROOT/'evidence/PREDICTIONS_COMPLETE.json',{'files':len(rows),'records':rows})
