"""CPU-only integration checks, real decode tests on labelled temporary fixtures."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU preparation must hide CUDA')
    code=subprocess.call([sys.executable,'-m','unittest','discover','-s','pipeline/tests','-v'],cwd=ROOT)
    if code:return code
    sys.path.insert(0,str(ROOT))
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA initialization')
    if 'pipeline.spine' in sys.modules:raise RuntimeError('CPU check imported native backend')
    from pipeline.entry import status
    from pipeline.guard import run_phase
    if status(ROOT)['gates']['paired']['allowed']:raise RuntimeError('actual inputs appeared; stop prep check rather than dispatch')
    old=subprocess.check_output
    def no_gpu_query(*args,**kw):raise AssertionError('missing-data path must not query/allocate GPU')
    subprocess.check_output=no_gpu_query
    try:refusal=run_phase('paired',ROOT)
    finally:subprocess.check_output=old
    assert refusal['stage']=='REFUSED_BEFORE_GPU' and refusal['gpu_device_seconds']==0
    result={'status':'CPU_PREPARATION_CHECKS_PASS','actual_recordings_received':0,'synthetic_fixtures_only':True,
            'native_model_imported':False,'cuda_initialized':False,'gpu_device_seconds':0,'cloud_calls':0,
            'missing_data_refusal':refusal,'empirical_measurement_calibrated':False,'native_daily_renderer_tested':False,
            'native_daily_training_tested':False,'no_method_results':True}
    dest=ROOT/'prep-stage/CPU_CHECKS.json';dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':sys.exit(main())
