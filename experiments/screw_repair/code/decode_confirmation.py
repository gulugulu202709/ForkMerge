import numpy as np
from common import ROOT,config,dump,sha
from render import decode

def build():
    records=[]
    for draw in config()['draws']:
      for split in config()['test_splits']:
        p=ROOT/'data'/f'draw{draw}'/(split+'.npz')
        with np.load(p) as d:rgb=d['rgb'];y=d['targets'];ids=d['family_ids']
        shape=rgb.shape[:3];frames=rgb.reshape(-1,*rgb.shape[3:])
        z=np.stack([decode(v) for v in frames]).astype(np.float32)/np.array([1000,200,1.6,300],np.float32)
        target=ROOT/'features/decoded'/f'draw{draw}'/(split+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists();np.savez_compressed(target,x=z.reshape(*shape,4),targets=y,family_ids=ids)
        records.append({'path':str(target.relative_to(ROOT)),'sha256':sha(target),'source_sha256':sha(p)})
    dump(ROOT/'features/decoded/MANIFEST.json',{'records':records,'source':'actual public RGB seven-segment decode, no private-state inputs'})
if __name__=='__main__':build()
