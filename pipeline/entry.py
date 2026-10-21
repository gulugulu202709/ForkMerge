"""One entrypoint: CPU preparation/review; explicit gated GPU phases later."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]

def status(root=ROOT):
    from pipeline.policy import phase_gate,read_json
    source=root/'pipeline_data/SOURCE_READY.json'
    measurement=root/'pipeline_state/MEASUREMENT_READY.json'
    gates={p:phase_gate(root,p) for p in ('diagnostics','cache','paired','paired_worst_witness')}
    outputs={arm:read_json(root/'pipeline_results'/arm/'COMPLETE.json',{}) for arm in ('paired','paired_worst_witness')}
    return {'stage':'WAITING_RECORDINGS' if not source.exists() else 'SOURCE_PRESENT_FURTHER_GATES_REQUIRED',
            'source_ready':bool(read_json(source,{}).get('calibration_ready')),
            'measurement_ready':bool(read_json(measurement,{}).get('ready')),
            'gpu_started':any((root/'pipeline_state/gpu').glob('*_started.json')),
            'training_prerequisites_met':gates['paired']['allowed'],
            'launch_permission':'ONLY_GUARD_CAN_CHECK_CURRENT_BUDGET_DEVICE_AND_ONCE_ONLY_ADMISSION',
            'method_results_available':any(record.get('complete') is True for record in outputs.values()),
            'method_output_status':{a:{'complete':r.get('complete',False),'reason':r.get('reason','NOT_RUN')} for a,r in outputs.items()},
            'gates':gates,'source':read_json(source),'measurement':read_json(measurement)}

def main():
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['status','prepare','calibrate','review','run','report'])
    parser.add_argument('--phase',choices=['diagnostics','cache','paired','paired_worst_witness']);args=parser.parse_args()
    if args.command=='status':result=status()
    elif args.command=='report':
        from pipeline.report import render
        result=render(ROOT)
    elif args.command=='prepare':
        from pipeline.intake import prepare
        result=prepare(ROOT)
    elif args.command=='calibrate':
        from pipeline.measurement import calibrate
        result=calibrate(ROOT)
    elif args.command=='review':
        if args.phase not in ('diagnostics','paired','paired_worst_witness'):parser.error('review needs --phase')
        from pipeline.measurement import apply_review
        result=apply_review(ROOT,args.phase)
    else:
        if not args.phase:parser.error('run needs --phase')
        from pipeline.guard import run_phase
        result=run_phase(args.phase)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
