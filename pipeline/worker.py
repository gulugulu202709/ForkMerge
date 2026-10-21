"""Real native video worker. Only a settled-budget guard may invoke main()."""
import argparse,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import config as C
from .model import OrdinaryUpdater,paired_objective,trace_statistics,witness_masks
from .policy import phase_gate,read_json

def put(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def event(root,stage,**kw):
    row={'stage':stage,'epoch':time.time(),**kw};state=root/'pipeline_state';state.mkdir(exist_ok=True)
    with (state/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    put(state/'worker_status.json',row);print(json.dumps(row),flush=True)

def pixels(rgb,device='cuda'):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(3,0,1,2).float().to(device)/127.5-1

def source_arrays(root,clip):
    return np.load(root/'pipeline_data/clips'/f'{clip["id"]}.npz')

def native_generate(root,pipe,reader,clip,memory,out,seed,oracle_end=None):
    from wan.utils.utils import cache_video
    z=source_arrays(root,clip);current=z['current_rgb'];width,height=C.SIZE
    source=torch.zeros(3,C.FRAMES,height,width,device='cuda')
    source[:,0]=pixels(current[None])[:,0];mask=torch.ones_like(source);mask[:,0]=0
    if oracle_end is not None:source[:,-1]=pixels(oracle_end[None])[:,0];mask[:,-1]=0
    out.parent.mkdir(parents=True,exist_ok=True);reader.memory=memory
    put(out.with_suffix('.input.json'),{'kind':'ORACLE_RENDERER_DIAGNOSTIC' if oracle_end is not None else 'NATIVE_FUTURE_FORECAST',
        'clip_id':clip['id'],'scenario':clip['scenario'],'seed':seed,'oracle_endpoint':oracle_end is not None,
        'condition':'current_image + actual_learned_memory' if memory is not None else 'current_image + truth_endpoint_diagnostic_only',
        'memory_shape':list(memory.shape) if memory is not None else None,'prompt':C.PROMPTS[clip['scenario']],
        'size':C.SIZE,'frames':C.FRAMES,'fps':C.FPS,'steps':C.STEPS,'CFG':C.CFG,'shift':C.SHIFT})
    with torch.no_grad():
        v=pipe.generate(C.PROMPTS[clip['scenario']],[source],[mask],[None],size=C.SIZE,frame_num=C.FRAMES,
            context_scale=1.,shift=C.SHIFT,sample_solver='unipc',sampling_steps=C.STEPS,guide_scale=C.CFG,
            seed=seed,offload_model=True,n_prompt=C.NEGATIVE)
    torch.save(v.detach().cpu().half(),out.with_suffix('.tensor.pt'))
    cache_video(tensor=v[None],save_file=str(out),fps=C.FPS,nrow=1,normalize=True,value_range=(-1,1));reader.memory=None

def diagnostics(root,pipe,reader,manifest):
    from .measurement import build_review
    samples=[]
    for i,scenario in enumerate(C.SCENARIOS):
        index={'cup':2,'teapot':1,'candy':0}[scenario]
        clip=sorted((c for c in manifest['clips'] if c['split']=='calibration' and c['scenario']==scenario),key=lambda c:c['id'])[index]
        path=root/'pipeline_results/diagnostics'/f'{scenario}_ORACLE_RENDERER_ONLY.mp4'
        native_generate(root,pipe,reader,clip,None,path,C.SEEDS[0]+i,source_arrays(root,clip)['future_rgb'][-1])
        samples.append({'sample_id':f'diagnostic_{scenario}','video':str(path),'scenario':scenario,'clip_id':clip['id']})
        event(root,'DIAGNOSTIC_VIDEO_SAVED',scenario=scenario,path=str(path),not_method_result=True)
    build_review(root,'diagnostics',samples)
    put(root/'pipeline_results/diagnostics/COMPLETE.json',{'samples':samples,'kind':'PRIVILEGED_RENDERER_DIAGNOSTIC_NOT_METHOD_RESULT'})
    event(root,'WAITING_BLIND_DIAGNOSTIC_REVIEW',gpu_released_on_exit=True)

def cache(root,pipe,reader,manifest):
    from .spine import first_image_context
    dest=root/'pipeline_cache';dest.mkdir(exist_ok=False);pipe.model.to('cuda');reader.memory=None
    with torch.no_grad():
        texts={s:t.cpu() for s,t in zip(C.SCENARIOS,pipe.text_encoder([C.PROMPTS[s] for s in C.SCENARIOS],torch.device('cpu')))}
        torch.save(texts,dest/'texts.pt')
        for clip in [c for c in manifest['clips'] if c['split']!='calibration']:
            z=source_arrays(root,clip);features=[]
            for stage in z['history_rgb']:
                rgb=pixels(stage)
                small=F.interpolate(rgb.permute(1,0,2,3),size=(128,224),mode='bilinear',align_corners=False).permute(1,0,2,3)
                latent=pipe.vae.encode([small])[0]
                with torch.autocast('cuda',dtype=torch.bfloat16):patch=pipe.model.patch_embedding(latent[None])
                features.append(F.adaptive_avg_pool3d(patch.float(),(4,4,4)).flatten(2).transpose(1,2)[0].cpu())
            vace,_=first_image_context(pipe,pixels(z['current_rgb'][None]),frames=C.FRAMES)
            payload={'features':torch.stack(features),'actions':torch.tensor([C.ACTIONS[a] for a in clip['actions']]),
                     'vace':[x.cpu() for x in vace],'id':clip['id'],'scenario':clip['scenario'],'split':clip['split']}
            if clip['split'] in ('train','dev'):payload['future']=pipe.vae.encode([pixels(z['future_rgb'])])[0].cpu()
            torch.save(payload,dest/f'{clip["id"]}.pt');event(root,'COMMON_CACHE_CLIP',clip_id=clip['id'],future_encoded=clip['split'] in ('train','dev'))
    put(dest/'READY.json',{'ready':True,'clip_ids':[c['id'] for c in manifest['clips'] if c['split']!='calibration'],
                         'test_future_encoded':False,'shared_for_both_arms':True})

def load_cached(root,clip_id):return torch.load(root/'pipeline_cache'/f'{clip_id}.pt',map_location='cpu',weights_only=True)

def flow_forward(pipe,reader,memory,z,t,text,vace):
    reader.memory=memory
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return pipe.model([z],t=t,vace_context=[x.to('cuda') for x in vace],context=[text],seq_len=math.prod(z.shape[1:])//4)[0].float()

def unit_loss(root,pipe,reader,model,unit,clips,texts,seed,step,arm,zero=False):
    meta=[clips[c] for c in unit['clip_ids']];data=[load_cached(root,c['id']) for c in meta]
    memory=[model(d['features'][None].to('cuda'),d['actions'][None].to('cuda')) for d in data]
    if zero:memory=[m*0 for m in memory]
    rng=torch.Generator(device='cuda').manual_seed(seed*10000+step);sigma=float(torch.rand((),device='cuda',generator=rng)*.7+.2)
    noise=torch.randn(data[0]['future'].shape,device='cuda',generator=rng);preds=[];targets=[]
    for d,m in zip(data,memory):
        future=d['future'].to('cuda').float()
        if future.shape!=noise.shape:raise ValueError('unmatched target latent shape')
        preds.append(flow_forward(pipe,reader,m,(1-sigma)*future+sigma*noise,torch.tensor([sigma*1000],device='cuda'),texts[unit['scenario']],d['vace']))
        targets.append(noise-future)
    return paired_objective(torch.stack(preds),torch.stack(targets),witness_masks(meta,data[0]['future'].shape,'cuda'),arm)

def train_seed(root,pipe,reader,manifest,arm,seed,texts):
    pipe.model.to('cuda');torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=OrdinaryUpdater().to('cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=C.LR,weight_decay=.01)
    clips={c['id']:c for c in manifest['clips']};units=sorted((u for u in manifest['units'] if u['split']=='train'),key=lambda u:u['id'])
    fixed=[next(u for u in units if u['scenario']==s) for s in C.SCENARIOS]
    def probes(choices,zero=False):
        values=[]
        with torch.no_grad():
            for u in choices:
                _,stats=unit_loss(root,pipe,reader,model,u,clips,texts,seed,900,arm,zero);values.append({'unit':u['id'],**stats})
        return {'proper_flow':sum(x['proper_flow'] for x in values)/len(values),'per_unit':values}
    initial=probes(fixed);zero=probes(fixed,True);history=[]
    for step in range(C.UPDATES):
        opt.zero_grad(set_to_none=True);unit=units[step%len(units)]
        loss,stats=unit_loss(root,pipe,reader,model,unit,clips,texts,seed,step,arm)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm) or float(norm)<=0:raise RuntimeError('no finite nonzero native memory gradient')
        if not all(p.grad is None for p in pipe.model.parameters()):raise RuntimeError('generator is not frozen')
        opt.step();history.append({'update':step+1,'unit':unit['id'],**stats,'grad_norm':float(norm)});event(root,'TRAIN_UPDATE',arm=arm,seed=seed,**history[-1])
    final=probes(fixed);dev=probes(sorted((u for u in manifest['units'] if u['split']=='dev'),key=lambda u:u['id']))
    adequate=final['proper_flow']<=.95*min(initial['proper_flow'],zero['proper_flow'])
    out=root/'pipeline_results'/arm/'fits'/str(seed);out.mkdir(parents=True,exist_ok=False);torch.save(model.state_dict(),out/'final.pt')
    record={'seed':seed,'arm':arm,'initial':initial,'zero_memory':zero,'final':final,'dev':dev,'training_adequate':adequate,
        'updates':len(history),'history':history,'parameters':sum(p.numel() for p in model.parameters()),
        'claim_limit':'Finitebudget5percenttrainprobe gate does not establish global convergence or strongest possible baseline.'}
    put(out/'training.json',record);reader.memory=None;return model,record

def run_arm(root,pipe,reader,manifest,arm):
    from .measurement import build_review
    texts={s:t.to('cuda') for s,t in torch.load(root/'pipeline_cache/texts.pt',map_location='cpu',weights_only=True).items()};reports=[];samples=[]
    tests=sorted((c for c in manifest['clips'] if c['split']=='test'),key=lambda c:c['id'])
    roster=[{'sample_id':f'{arm}_{seed}_{c["id"]}','clip_id':c['id'],'scenario':c['scenario'],'seed':seed,'status':'NOT_DISPATCHED'} for seed in C.SEEDS for c in tests]
    put(root/'pipeline_results'/arm/'ROSTER.json',roster)
    for seed in C.SEEDS:
        model,record=train_seed(root,pipe,reader,manifest,arm,seed,texts);reports.append(record)
        if not record['training_adequate']:
            put(root/'pipeline_results'/arm/'COMPLETE.json',{'training_adequate':False,'reports':reports,'complete':False,'reason':'UNDERFIT_INCONCLUSIVE'})
            event(root,'STOP_INCONCLUSIVE_TRAINING',arm=arm,seed=seed);return
        for i,clip in enumerate(tests):
            d=load_cached(root,clip['id'])
            with torch.no_grad():m,states=model(d['features'][None].to('cuda'),d['actions'][None].to('cuda'),True)
            out=root/'pipeline_results'/arm/'videos'/str(seed)/f'{clip["id"]}.mp4';out.parent.mkdir(parents=True,exist_ok=True)
            memory_path=out.with_suffix('.memory.pt')
            torch.save({'kind':'ACTUAL_TRAINED_UPDATER_MEMORY','memory':m.detach().cpu(),'states':[x.detach().cpu() for x in states],
                'actions':d['actions'],'clip_id':clip['id'],'seed':seed,'arm':arm},memory_path)
            put(out.with_suffix('.memory.json'),trace_statistics(m,states));native_generate(root,pipe,reader,clip,m,out,seed*100+i)
            sample={'sample_id':f'{arm}_{seed}_{clip["id"]}','video':str(out),'scenario':clip['scenario'],'clip_id':clip['id'],'arm':arm,'seed':seed,'memory':str(memory_path)}
            samples.append(sample)
            for row in roster:
                if row['sample_id']==sample['sample_id']:row.update(status='SAVED_PENDING_BLIND_REVIEW',video=str(out))
            put(root/'pipeline_results'/arm/'ROSTER.json',roster);event(root,'NATIVE_FORECAST_SAVED',**sample)
        del model;torch.cuda.empty_cache()
    build_review(root,arm,samples)
    put(root/'pipeline_results'/arm/'COMPLETE.json',{'training_adequate':True,'complete':True,'reports':reports,'samples':samples,
        'accuracy':None,'reason':'WAITING_BLIND_REVIEW','memory_meaning':'Actual tensors, not a semantic erasure proof.'})
    event(root,'WAITING_BLIND_METHOD_REVIEW',arm=arm,gpu_released_on_exit=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['diagnostics','cache',*C.METHODS]);args=parser.parse_args();root=C.ROOT
    if os.environ.get('FORKMERGE_BUDGET_PHASE')!=args.phase:raise RuntimeError('Budget guard required')
    gate=phase_gate(root,args.phase)
    if not gate['allowed']:raise RuntimeError(gate['reason'])
    manifest=read_json(root/'pipeline_data/manifest.json')
    from .spine import load_pipe,Reader
    pipe=load_pipe();reader=Reader(pipe)
    if args.phase=='diagnostics':diagnostics(root,pipe,reader,manifest)
    elif args.phase=='cache':cache(root,pipe,reader,manifest)
    else:run_arm(root,pipe,reader,manifest,args.phase)
if __name__=='__main__':main()
