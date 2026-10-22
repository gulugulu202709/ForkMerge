import json
from pathlib import Path
import numpy as np
from common import ROOT,config,dump,sha
from family_source import parameters,family
from physics import outcomes,reference_from_visible,decisions

def build():
    cfg=config();source=Path(cfg['source_root']);dest=ROOT/'data';dest.mkdir(exist_ok=False)
    old=set();rows=[]
    for p in sorted((source/'data').glob('draw*/*_audit.npz')):
        with np.load(p) as d:old.update(map(tuple,d['parameters'].tolist()))
    seen=set(old);new=[]
    for draw in cfg['draws']:
        rng=np.random.default_rng(cfg['confirmation_data_seed']+draw)
        pars=[parameters(rng,cfg['test_stiffness_N_mm'],seen,test=True) for _ in range(cfg['confirmation_families_per_draw'])]
        assert not old.intersection(pars);assert not set(new).intersection(pars);new.extend(pars)
        for split in cfg['test_splits']:
            ops=[split if split in cfg['heldout_operations'] else ('retighten' if i%2==0 else 'replace_interface') for i in range(len(pars))]
            items=[family(par,op,rng) for par,op in zip(pars,ops)]
            rgb=np.stack([v[0] for v in items]);decoded=np.stack([v[1] for v in items]);states=np.stack([v[2] for v in items])
            y=outcomes(states).reshape(len(pars),4,9);ref,_=reference_from_visible(decoded);ref=ref.reshape(y.shape)
            assert np.abs(ref-y).mean()<.005
            approve,chosen=decisions(y,cfg);ra,rc=decisions(ref,cfg)
            assert np.array_equal(approve,ra) and np.array_equal(chosen,rc)
            ids=np.arange(len(pars))+100000*(draw+1)
            p=dest/f'draw{draw}'/(split+'.npz');p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,rgb=rgb,targets=y.astype(np.float32),family_ids=ids)
            audit=p.with_name(split+'_audit.npz')
            np.savez_compressed(audit,decoded=decoded,states=states,reference=ref,parameters=np.asarray(pars),
                roles=np.stack([v[3] for v in items]),operations=np.asarray(ops),family_ids=ids)
            row={'draw':draw,'split':split,'families':len(pars),'pixel_reference_MAE_mm':float(np.abs(ref-y).mean()),
               'pixel_reference_max_mm':float(np.abs(ref-y).max()),'pixel_decision_agreement':1.,
               'unsafe_keep_cases':int((~approve).sum()),'unsafe_keep_families':int((~approve).any((1,2)).sum()),
               'public_sha256':sha(p),'audit_sha256':sha(audit)}
            rows.append(row);print('CONFIRMATION_DATA',row,flush=True)
    dump(dest/'MANIFEST.json',{'panels':rows,'new_base_families':len(new),'excluded_original_families':len(old),
       'no_base_overlap_with_any_original_split':True,'panels_matched_not_independent':True,
       'draw_sampling':'sequential without replacement, disjoint but coupled','protocol_sha256':sha(ROOT/'configs/protocol.json')})
if __name__=='__main__':build()
