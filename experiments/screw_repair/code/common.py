import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(os.environ.get("FORKMERGE_SCREW_RUN", Path(__file__).resolve().parents[1]))
PYTHON = sys.executable
PAIRS = [(a,b) for a in range(4) for b in range(a+1,4)]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp,path)

def config():
    cfg = json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8'))
    cfg['source_root'] = str(Path(os.environ.get('FORKMERGE_SCREW_SOURCE', ROOT/'external_source')))
    cfg['siglip_weight_dir'] = str(Path(os.environ.get('FORKMERGE_SIGLIP_WEIGHTS', Path(cfg['source_root'])/'weights/siglip2')))
    return cfg

def code_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'code').glob('*.py'))}
