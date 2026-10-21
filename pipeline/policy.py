"""CPU-only source/review gating and settled-budget arithmetic."""
import json
from pathlib import Path
from .config import ARM_CAP_SECONDS, PRIOR_GPU_SECONDS, TOTAL_GPU_SECONDS

def read_json(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default

def observed_pattern_errors(manifest):
    clips={c['id']:c for c in manifest.get('clips',[])};groups=[];errors=[]
    for scenario in ('cup','teapot','candy'):
        rows=[c for c in clips.values() if c.get('split')=='calibration' and c.get('scenario')==scenario]
        groups.append(('calibration_'+scenario,scenario,rows))
    for unit in manifest.get('units',[]):
        if unit.get('split')!='calibration':groups.append((unit['id'],unit['scenario'],[clips[x] for x in unit['clip_ids']]))
    def value(v):return ' '.join(v.strip().lower().split()) if isinstance(v,str) else v
    for name,scenario,rows in groups:
        if len(rows)!=3:errors.append(name+': requires3actual traces');continue
        try:
            left=[value(c['readouts']['left']['truth']) for c in rows]
            right=[value(c['readouts']['right']['truth']) for c in rows]
            if any(v is None for v in left+right):raise ValueError('unobserved truth')
            if scenario in ('cup','candy'):
                okay=len(set(left))==1 and len(set(right))==2
            else:
                pairs=[(0,1),(0,2),(1,2)]
                okay=any(left[i]==left[j] and right[i]!=right[j] for i,j in pairs) and any(right[i]==right[j] and left[i]!=left[j] for i,j in pairs)
            if not okay:errors.append(name+': actual outcomes do not instantiate intended erase/protect contrasts')
        except (KeyError,TypeError,ValueError):errors.append(name+': missing actual readouts')
    return errors

def remaining_budget(events,prior=PRIOR_GPU_SECONDS):
    open_stages={};used=float(prior);arm_used={'paired':0.,'paired_worst_witness':0.}
    for row in events:
        stage=row['phase']
        if row['event']=='start':
            if stage in open_stages:raise ValueError('duplicate stage start')
            open_stages[stage]=row
        elif row['event']=='end':
            if stage not in open_stages:raise ValueError('end without start')
            del open_stages[stage]
            charge=float(row['gpu_device_seconds'])
            if charge<0:raise ValueError('negative device charge')
            used+=charge
            if stage in arm_used:arm_used[stage]+=charge
    if open_stages:raise ValueError('Unsettled GPU segment: manual reconciliation required, never automatic retry')
    return {'used_seconds':used,'remaining_seconds':max(0.,TOTAL_GPU_SECONDS-used),'arm_used':arm_used,
            'arm_remaining':{a:max(0.,ARM_CAP_SECONDS-v) for a,v in arm_used.items()}}

def phase_gate(root,phase):
    root=Path(root);data=root/'pipeline_data';state=root/'pipeline_state'
    if phase not in ('diagnostics','cache','paired','paired_worst_witness'):return {'allowed':False,'reason':'UNKNOWN_PHASE'}
    source=read_json(data/'SOURCE_READY.json',{})
    if not source.get('calibration_ready'):return {'allowed':False,'reason':'WAITING_VERIFIED_RECORDINGS'}
    pattern_errors=observed_pattern_errors(read_json(data/'manifest.json',{}))
    if pattern_errors:return {'allowed':False,'reason':'SOURCE_CONTRASTS_NOT_VERIFIED','errors':pattern_errors}
    measure=read_json(state/'MEASUREMENT_READY.json',{})
    if not measure.get('ready'):return {'allowed':False,'reason':'WAITING_MEASUREMENT_CALIBRATION'}
    if phase=='diagnostics':return {'allowed':True,'reason':'CALIBRATION_ONLY_RENDERER_CEILING'}
    if not source.get('formal_ready'):return {'allowed':False,'reason':'WAITING_DISJOINT_FORMAL_DATA'}
    diag=read_json(state/'reviews/diagnostics/score.json',{})
    if not(diag.get('complete') and diag.get('judgeable') and diag.get('all_correct')):
        return {'allowed':False,'reason':'WAITING_OR_FAILED_RENDERER_REVIEW'}
    if phase=='cache':return {'allowed':True,'reason':'PREPARE_COMMON_FORMAL_CACHE'}
    if not (root/'pipeline_cache/READY.json').is_file():return {'allowed':False,'reason':'WAITING_SHARED_CACHE'}
    if phase=='paired':return {'allowed':True,'reason':'GATES_PRESENT_RUNTIME_REVALIDATION_REQUIRED'}
    training=read_json(root/'pipeline_results/paired/COMPLETE.json',{})
    if not training.get('training_adequate'):return {'allowed':False,'reason':'ORDINARY_UNFINISHED_OR_UNDERFIT'}
    ordinary=read_json(state/'reviews/paired/score.json',{})
    if not ordinary.get('complete'):return {'allowed':False,'reason':'WAITING_ORDINARY_REVIEW'}
    if not ordinary.get('judgeable'):return {'allowed':False,'reason':'INCONCLUSIVE_ORDINARY_EXPOSURE'}
    if ordinary.get('all_correct'):return {'allowed':False,'reason':'STOP_ORDINARY_SOLVES_PILOT'}
    return {'allowed':True,'reason':'ORDINARY_ADEQUATE_WITH_REVIEWED_ERRORS'}
