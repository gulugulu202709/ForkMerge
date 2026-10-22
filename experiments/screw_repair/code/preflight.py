import importlib.metadata,json
from pathlib import Path
from common import ROOT,config,dump,sha,code_hashes

def run():
    cfg=config();source=Path(cfg['source_root']);env=json.loads((source/'.aris/compute/env-spec.json').read_text())
    assert {p:importlib.metadata.version(p) for p in env['packages']}==env['packages']
    for p,h in env['weights'].items():assert sha(Path(p) if Path(p).is_absolute() else source/p)==h
    freeze=json.loads((source/'evidence/FIT_FREEZE.json').read_text());reuse=[];fits=[]
    for item in freeze['selected_checkpoints']:
        if item['budget']!=1200:continue
        visual=item['arm']=='siglip_transformer'
        decoded=item['arm']=='decoded_gru' and item['objective']=='paired'
        if not (visual or decoded):continue
        ck=source/item['path'];assert sha(ck)==item['sha256']
        row={**item,'arm':('tf_plain' if item['objective']=='plain' else 'tf_pair') if visual else 'decoded_pair',
             'path':str(ck),'origin':'reused_frozen_original','encoder':'siglip2' if visual else 'decoded'}
        reuse.append(row)
    assert len(reuse)==18
    for draw in cfg['draws']:
      for original,objective in [('siglip_transformer','plain'),('siglip_transformer','paired'),('decoded_gru','paired')]:
       for seed in cfg['seeds']:
        for lr in cfg['learning_rates']:
            d=source/'results'/f'draw{draw}'/f'{original}_{objective}_s{seed}_lr{lr:g}'
            info=json.loads((d/'FIT.json').read_text());assert info['epochs']==1200 and info['steps']==9600
            for p,h in info['inputs'].items():assert sha(source/p)==h
            assert sha(d/'best_1200.pt')==info['checkpoint_sha256']['1200']
            fits.append({'path':str(d/'FIT.json'),'sha256':sha(d/'FIT.json')})
    dump(ROOT/'evidence/REUSED_BASELINES.json',{'selected_checkpoints':reuse,'all36_candidate_fits':fits,
        'source_freeze_sha256':sha(source/'evidence/FIT_FREEZE.json')})
    dump(ROOT/'evidence/PRELAUNCH_FREEZE.json',{'protocol_sha256':sha(ROOT/'configs/protocol.json'),
      'code_hashes':code_hashes(),'environment_spec_sha256':sha(source/'.aris/compute/env-spec.json'),
      'new_fits':24,'reused_fits':36,'selected_total':30,'confirmation_predictions_expected':90,
      'scientific_scope':'simulation_only; continuous-response canonical-witness adaptation; not original count classifier or world model'})
    print('DIRECT_PREFLIGHT_PASS',len(reuse),len(fits),flush=True)
if __name__=='__main__':run()
