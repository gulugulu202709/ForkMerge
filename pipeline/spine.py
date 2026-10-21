"""Native VACE loader reused from prior witnessed spine. GPU worker imports only."""
from pathlib import Path
import os
import sys
import torch
from torch.utils.checkpoint import checkpoint
REPO=Path(os.environ.get('FORKMERGE_VACE_REPO',Path(__file__).resolve().parents[1]/'third_party/VACE'))
WEIGHTS=Path(os.environ.get('FORKMERGE_MODEL_DIR',Path(__file__).resolve().parents[1]/'models/Wan2.1-VACE-1.3B'))
sys.path[:0]=[str(REPO/'vace'),str(REPO)]
from models.wan import WanVace
from models.wan.configs import WAN_CONFIGS

class Reader:
    def __init__(self,pipe):
        self.memory=None
        self.hook=pipe.model.text_embedding.register_forward_hook(self.inject)
    def inject(self,module,args,output):
        if self.memory is None:return output
        assert self.memory.ndim==3 and tuple(self.memory.shape[1:])==(64,1536)
        return torch.cat([output,self.memory.to(output.dtype)],1)

def load_pipe():
    torch.set_num_threads(2)
    pipe=WanVace(WAN_CONFIGS['vace-1.3B'],str(WEIGHTS),device_id=0,t5_cpu=True)
    pipe.model.eval().requires_grad_(False);pipe.vae.model.eval().requires_grad_(False)
    for block in list(pipe.model.blocks)+list(pipe.model.vace_blocks):
        original=block.forward
        def wrapped(*args,_original=original,**kw):
            return checkpoint(_original,*args,use_reentrant=False,**kw) if torch.is_grad_enabled() else _original(*args,**kw)
        block.forward=wrapped
    return pipe

def first_image_context(pipe,rgb,frames=21):
    src=torch.zeros((3,frames,*rgb.shape[-2:]),device=pipe.device)
    src[:,0]=rgb[:,-1].to(pipe.device)
    mask=torch.ones_like(src);mask[:,0]=0
    with torch.no_grad():
        z0=pipe.vace_encode_frames([src],[None],masks=[mask])
        m=pipe.vace_encode_masks([mask],[None])
        context=pipe.vace_latent(z0,m)
    return context,(16,(frames-1)//4+1,src.shape[-2]//8,src.shape[-1]//8)
