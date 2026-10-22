import argparse,hashlib,time
import numpy as np
import torch
from common import ROOT,config,dump,sha
from feature_source import forward

def extract(draw,device='cuda'):
    from transformers import SiglipVisionModel,AutoImageProcessor
    torch.set_num_threads(2);torch.manual_seed(0);start=time.time();cfg=config()
    m,info=SiglipVisionModel.from_pretrained(cfg['siglip_weight_dir'],local_files_only=True,
        attn_implementation='eager',output_loading_info=True)
    assert not info['missing_keys'] and not info.get('mismatched_keys')
    m=m.to(device).eval();pp=AutoImageProcessor.from_pretrained(cfg['siglip_weight_dir'],local_files_only=True)
    dest=ROOT/'features/siglip2'/f'draw{draw}';dest.mkdir(parents=True,exist_ok=False);records=[]
    for split in cfg['test_splits']:
        source=ROOT/'data'/f'draw{draw}'/(split+'.npz')
        with np.load(source) as d:rgb=d['rgb'];target=d['targets'];ids=d['family_ids']
        shape=rgb.shape[:3];images=rgb.reshape(-1,*rgb.shape[3:]);mapping={};unique=[];inverse=[]
        for v in images:
            h=hashlib.sha256(v.tobytes()).digest()
            if h not in mapping:mapping[h]=len(unique);unique.append(v)
            inverse.append(mapping[h])
        parts=[]
        with torch.inference_mode():
            for i in range(0,len(unique),32):
                parts.append(forward(m,pp,np.asarray(unique[i:i+32]),device).cpu().numpy())
                if i%2048==0:print('ENCODE_CONFIRMATION',draw,split,i,len(unique),flush=True)
        z=np.concatenate(parts).astype(np.float32)[np.asarray(inverse)].reshape(*shape,-1)
        p=dest/(split+'.npz');np.savez_compressed(p,x=z,targets=target,family_ids=ids)
        records.append({'split':split,'source_sha256':sha(source),'output_sha256':sha(p),'shape':list(z.shape)})
        print('CONFIRMATION_FEATURES',draw,split,flush=True)
    dump(dest/'MANIFEST.json',{'records':records,'seconds':time.time()-start,'device':device,'trainable_backbone':False,
      'preprocessing':'same original dense2x2 SigLIP2 cache recipe, default extraction TF32 policy; no fitted transforms'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draw',type=int,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();extract(a.draw,a.device)
