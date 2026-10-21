"""Actual settled device costs, kept separate from scientific conclusions."""
import json
from pathlib import Path
from .config import PARENT,PRIOR_GPU_SECONDS,TOTAL_GPU_SECONDS,METHODS
from .policy import read_json,remaining_budget

def cost_summary(root):
    root=Path(root);ledger=PARENT/'evidence/everyday_continuation_events.jsonl'
    if ledger.exists():events=[json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    else:
        events=[];gpu=root/'pipeline_state/gpu'
        for start in sorted(gpu.glob('*_started.json')):
            row=read_json(start);events.append(row)
            end=gpu/f'{row["phase"]}_completion.json'
            if end.exists():events.append(read_json(end))
    result={'settled':True,'prior_gpu_seconds':PRIOR_GPU_SECONDS,'daily_gpu_seconds':0.,'shared_gpu_seconds':0.,
            'arm_gpu_seconds':{a:0. for a in METHODS},'allocated_total_per_completed_arm':{},'total_cap_seconds':TOTAL_GPU_SECONDS}
    for row in events:
        if row['event']!='end':continue
        seconds=float(row['gpu_device_seconds']);result['daily_gpu_seconds']+=seconds
        if row['phase'] in METHODS:result['arm_gpu_seconds'][row['phase']]+=seconds
        else:result['shared_gpu_seconds']+=seconds
    try:
        budget=remaining_budget(events);result['total_gpu_seconds']=budget['used_seconds'];result['remaining_gpu_seconds']=budget['remaining_seconds']
    except ValueError as exc:
        result.update(settled=False,total_gpu_seconds=None,remaining_gpu_seconds=None,reason=str(exc))
    completed=[a for a in METHODS if read_json(root/'pipeline_results'/a/'COMPLETE.json',{}).get('complete')]
    for arm in completed:result['allocated_total_per_completed_arm'][arm]=result['arm_gpu_seconds'][arm]+result['shared_gpu_seconds']/len(completed)
    result['comparison_complete']=len(completed)==2 and result['settled']
    result['allocation_note']='Shared diagnostics/cache divided equally across completed arms. Prior failed screw diagnostic remains separate original-budget spending. Partial arms do not support an efficacy or speed comparison.'
    return result
