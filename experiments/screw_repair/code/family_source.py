"""Fresh, disjoint base families; matched held-out repair panels.

Test panels deliberately share pre-repair families across the three operators.
They are not three independent samples. No training/validation/test or draw
shares a base parameter family. Model-facing files contain RGB and outcomes;
private operation names/parameters never enter the forward pass.
"""
import argparse, json
import numpy as np
from common import ROOT,config,dump,sha
from physics import repair,outcomes,reference_from_visible,damage,decisions
from render import render,decode

def parameters(rng,ks,seen,test=False):
    while True:
        k=float(rng.choice(ks)); p0,p1=sorted(map(float,rng.choice(np.arange(15,196,10),2,replace=False)))
        healthy=round(float(rng.uniform(.14,.19)),3)
        harmed=round(float(rng.uniform(.201,.72)),3)
        support=float(rng.choice([0,120,240] if test else [0,120,240,360,480]))
        key=(k,p0,p1,healthy,harmed,support)
        if key not in seen: seen.add(key); return key

def family(par,operation,rng):
    k,p0,p1,hpk,dpk=par[:5]; initialsupport=par[5] if len(par)>5 else 0.
    background=rng.integers(240,253,3)
    rgbs=[]; decs=[]; states=[]; roles=[]
    for kind in range(2):
      for route in range(2):
        peak=[hpk,dpk][kind]; oldp=[p0,p1][route]
        _,postp,postpk,support=map(float,repair(k,oldp,peak,initialsupport,operation))
        events=['calibrate','preload','load','load','unload','wait','repair','unload','probe','unload','probe','unload']
        rows=[]
        for t,event in enumerate(events):
            f,p,x,b=0.,0.,0.,initialsupport
            if t==0: f,p,x,b=(k+b)*.1,0.,.1,b
            elif t==1: p=oldp
            elif t in (2,3):
                x=.1 if t==2 else peak; p=oldp; f=p+((1-damage(max(.1,x)))*k+b)*x
            elif t in (4,5): p=oldp
            else:
                p,b=postp,support
                if t in (8,10):
                    x=.08 if t==8 else .12
                    f=p+((1-damage(postpk))*k+b)*x
            rows.append(render([f,p,x,b],event,background,x,b>0))
        rgb=np.asarray(rows); decoded=np.stack([decode(v) for v in rgb])
        rgbs.append(rgb); decs.append(decoded)
        states.append([k,postp,max(.12,postpk),support]); roles.append([kind,route])
    order=rng.permutation(4)
    rgb=np.asarray(rgbs)[order]; dec=np.asarray(decs)[order]; st=np.asarray(states)[order]
    assert all(np.array_equal(rgb[0,-1],rgb[b,-1]) for b in range(4))
    return rgb,dec,st,np.asarray(roles)[order],background

def build(smoke=False):
    cfg=config(); dest=ROOT/('smoke_data_v2' if smoke else 'data'); dest.mkdir(exist_ok=False)
    seen=set(); manifests=[]
    for draw in cfg['draws']:
        rng=np.random.default_rng(cfg['data_seed']+draw)
        sharedtest=[parameters(rng,cfg['test_stiffness_N_mm'],seen,test=True) for _ in range(8 if smoke else cfg['split_sizes']['test'])]
        panels={'train':None,'validation':None,'seen_repair':sharedtest,'partial_patch':sharedtest,'brace_install':sharedtest}
        for split,shared in panels.items():
            if shared is None:
                n=8 if smoke else cfg['split_sizes'][split]
                pars=[parameters(rng,cfg['train_stiffness_N_mm'],seen) for _ in range(n)]
            else: pars=shared
            items=[]; operations=[]
            for fi,par in enumerate(pars):
                op=split if split in cfg['heldout_operations'] else ('retighten' if fi%2==0 else 'replace_interface')
                items.append(family(par,op,rng)); operations.append(op)
            rgb=np.stack([v[0] for v in items]); decoded=np.stack([v[1] for v in items]); states=np.stack([v[2] for v in items])
            y=outcomes(states).reshape(len(pars),4,9)
            ref,rs=reference_from_visible(decoded); ref=ref.reshape(y.shape)
            err=np.abs(ref-y); assert err.mean()<cfg['oracle_mae_max_mm'],(split,float(err.mean()))
            approve,chosen=decisions(y,cfg);refapprove,refchosen=decisions(ref,cfg)
            agreement=float(np.mean(approve==refapprove));repairagreement=float(np.mean(chosen==refchosen))
            assert agreement>=.995 and repairagreement>=.995,(split,agreement,repairagreement)
            offset=0 if split=='train' else (1000 if split=='validation' else 2000)
            ids=np.arange(len(pars))+draw*10000+offset
            targetmargin=np.abs(y.reshape(len(pars),4,3,3)-cfg['decision']['opening_limit_mm'])
            public=dest/f'draw{draw}'/f'{split}.npz'; public.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(public,rgb=rgb,targets=y.astype(np.float32),family_ids=ids)
            private=public.with_name(split+'_audit.npz')
            np.savez_compressed(private,decoded=decoded,states=states,reference=ref,parameters=np.asarray(pars),
                roles=np.stack([v[3] for v in items]),operations=np.asarray(operations),family_ids=ids)
            row={'draw':draw,'split':split,'families':len(pars),'pixel_reference_MAE_mm':float(err.mean()),
                'pixel_reference_max_mm':float(err.max()),'robust_decision_margin_fraction':float((targetmargin>.04).mean()),
                'operation_counts':{op:operations.count(op) for op in set(operations)},
                'unsafe_keep_cases':int((~approve).sum()),'oracle_repair_action_counts':np.bincount(chosen.ravel(),minlength=3).tolist(),
                'pixel_reference_approval_agreement':agreement,'pixel_reference_repair_agreement':repairagreement,
                'public_sha256':sha(public),'audit_sha256':sha(private)}
            manifests.append(row); print('DATA_BUILT',row,flush=True)
    dump(dest/'MANIFEST.json',{'panels':manifests,'base_families_unique':len(seen),
        'test_panels_matched_not_independent':True,'heldout_operations_absent_from_training':True,
        'draw_sampling':'sequential without replacement, disjoint but coupled',
        'length':12,'oracle_scope':'same-law observability control, not independent physical validation'})
    return manifests

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();build(a.smoke)
