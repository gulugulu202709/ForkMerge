import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,config,dump,sha
from render import decode

def load_encoder(name,device):
    cfg=config()
    if name=='resnet18':
        from torchvision.models import resnet18
        m=resnet18(weights=None); m.load_state_dict(torch.load(cfg['resnet_weight'],map_location='cpu',weights_only=True))
        m=torch.nn.Sequential(*list(m.children())[:-2])
        return m.to(device).eval(),None
    if name=='siglip2':
        from transformers import SiglipVisionModel,AutoImageProcessor
        path=str(ROOT/'weights/siglip2')
        m,info=SiglipVisionModel.from_pretrained(path,local_files_only=True,attn_implementation='eager',output_loading_info=True)
        assert not info['missing_keys'],info
        assert not info.get('mismatched_keys'),info
        pp=AutoImageProcessor.from_pretrained(path,local_files_only=True)
        return m.to(device).eval(),pp
    raise ValueError(name)

def forward(m,pp,rgb,device):
    if pp is None:
        x=torch.from_numpy(np.asarray(rgb)).permute(0,3,1,2).to(device).float()/255
        x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
        mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        x=(x-mean)/std
        return F.adaptive_avg_pool2d(m(x),(2,2)).flatten(1)
    x=pp(images=list(rgb),return_tensors='pt')['pixel_values'].to(device)
    z=m(pixel_values=x).last_hidden_state
    h=int(z.shape[1]**.5); assert h*h==z.shape[1]
    z=z.reshape(len(z),h,h,-1).permute(0,3,1,2)
    return F.adaptive_avg_pool2d(z,(2,2)).flatten(1)

def extract(name,device='cpu',smoke=False):
    start=time.time();torch.set_num_threads(2);torch.manual_seed(0)
    dest=ROOT/('smoke_features' if smoke else 'features')/name;dest.mkdir(parents=True,exist_ok=False)
    data=ROOT/('smoke_data_v2' if smoke else 'data');model=pp=None
    if name!='decoded': model,pp=load_encoder(name,device)
    records=[]
    for source in sorted(data.glob('draw*/*.npz')):
        if source.name.endswith('_audit.npz'):continue
        with np.load(source) as d:
            rgb=d['rgb'];target=d['targets'];ids=d['family_ids']
        shape=rgb.shape[:3]; images=rgb.reshape(-1,*rgb.shape[3:]); parts=[]
        if name=='decoded':
            z=np.stack([decode(v) for v in images]).astype(np.float32)
            z/=np.array([1000,200,1.6,300],np.float32)
        else:
            # Cache repeated images within each split; no train/test fitted transforms.
            import hashlib
            mapping={};unique=[];inverse=[]
            for v in images:
                h=hashlib.sha256(v.tobytes()).digest()
                if h not in mapping: mapping[h]=len(unique);unique.append(v)
                inverse.append(mapping[h])
            with torch.inference_mode():
                for i in range(0,len(unique),32):
                    parts.append(forward(model,pp,np.asarray(unique[i:i+32]),device).cpu().numpy())
                    if i%1024==0:print('ENCODE',name,source.name,i,len(unique),flush=True)
            u=np.concatenate(parts).astype(np.float32);z=u[np.asarray(inverse)]
        outfile=dest/source.relative_to(data);outfile.parent.mkdir(exist_ok=True)
        np.savez_compressed(outfile,x=z.reshape(*shape,-1),targets=target,family_ids=ids)
        records.append({'input':str(source.relative_to(ROOT)),'input_sha256':sha(source),'output':str(outfile.relative_to(ROOT)),
            'output_sha256':sha(outfile),'shape':list(z.reshape(*shape,-1).shape)})
        print('FEATURES',name,str(source),flush=True)
    dump(dest/'MANIFEST.json',{'encoder':name,'records':records,'device':device,'seconds':time.time()-start,
        'trainable_backbone':False,'input_scope':'all public RGB frames, no future RGB/private states'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encoder',required=True);p.add_argument('--device',default='cpu');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();extract(a.encoder,a.device,a.smoke)
