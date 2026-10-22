"""Bounded later-stage continuation. CPU prerequisites run before any allocation."""
import argparse,ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path
import shutil
from . import config as C
from .policy import phase_gate,read_json,remaining_budget

def budget_events(ledger):
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []

def assert_frozen_source(root):
    dest=root/'pipeline_state/frozen_code'
    files=sorted((root/'pipeline').glob('*.py'))
    if dest.exists():
        if {p.name for p in files}!={p.name for p in dest.glob('*.py')}:raise RuntimeError('pipeline source file set changed after diagnostics')
        if any(p.read_bytes()!=(dest/p.name).read_bytes() for p in files):raise RuntimeError('pipeline source changed after diagnostics')
    else:
        dest.mkdir(parents=True)
        for path in files:shutil.copy2(path,dest/path.name)

def validate_inputs(root,phase):
    from .intake import prepare
    from .measurement import calibrate,apply_review
    prepare(root);calibrate(root)
    if phase!='diagnostics':apply_review(root,'diagnostics')
    if phase=='paired_worst_witness':apply_review(root,'paired')
    return phase_gate(root,phase)

def run_phase(phase,root=C.ROOT):
    root=Path(root)
    # Real current videos/measurements, not marker existence alone.
    gate=validate_inputs(root,phase)
    if not gate['allowed']:
        return {'stage':'REFUSED_BEFORE_GPU','phase':phase,**gate,'gpu_device_seconds':0}
    if sys.platform!='linux':
        return {'stage':'REFUSED_BEFORE_GPU','phase':phase,'allowed':False,
                'reason':'NATIVE_LAUNCH_REQUIRES_LINUX','gpu_device_seconds':0}
    import fcntl
    if str(root)!=str(C.ROOT):raise RuntimeError('native root cannot be a synthetic test directory')
    parent=C.PARENT;prior=budget_events(parent/'evidence/later_learning_events.jsonl')
    if len(prior)!=2 or prior[0]['event']!='start' or prior[1]['event']!='end' or prior[1]['exit_code']!=0:
        raise RuntimeError('prior budget is not settled')
    if abs(float(prior[1]['gpu_device_seconds'])-C.PRIOR_GPU_SECONDS)>1e-6:raise RuntimeError('prior charge changed')
    lock=(parent/'evidence/later_learning.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=parent/'evidence/everyday_continuation_events.jsonl';events=budget_events(ledger)
    settled=remaining_budget(events,float(prior[1]['gpu_device_seconds']))
    if any(e.get('phase')==phase for e in events):raise RuntimeError('phase already attempted; no automatic rerun')
    cap=min(settled['remaining_seconds'],settled['arm_remaining'][phase] if phase in C.METHODS else 450.)
    if cap<30:raise RuntimeError('insufficient remaining budget; no launch')
    occupancy=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(occupancy.strip())>=500:return {'stage':'DEVICE_BUSY_NO_LAUNCH','phase':phase,'gpu_device_seconds':0}
    assert_frozen_source(root)
    state=root/'pipeline_state';state.mkdir(exist_ok=True)
    if phase=='cache':
        frozen=state/'formal_frozen.json'
        if frozen.exists():raise RuntimeError('formal corpus already frozen by a previous attempt')
        frozen.write_text((root/'pipeline_data/manifest.json').read_text())
    if phase in C.METHODS:
        if read_json(state/'formal_frozen.json')!=read_json(root/'pipeline_data/manifest.json'):raise RuntimeError('formal corpus changed')
        (state/'fit_started.json').write_text(json.dumps({'phase':phase,'epoch':time.time()}))
    evidence=state/'gpu';evidence.mkdir(exist_ok=True)
    begin={'event':'start','phase':phase,'epoch':time.time(),'cap_seconds':cap,'owner_root':str(root),'gpu':1}
    start=time.monotonic()
    def append(row):
        with ledger.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    append(begin);(evidence/f'{phase}_started.json').write_text(json.dumps(begin,indent=2))
    flag=[False]
    for sig in (signal.SIGHUP,signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda n,f:flag.__setitem__(0,True))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',FORKMERGE_BUDGET_PHASE=phase,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    reason='child_exit';code=125;worker=None
    guard_pid=os.getpid()
    def pdeath():
        if ctypes.CDLL(None).prctl(1,signal.SIGKILL)!=0 or os.getppid()!=guard_pid:os._exit(125)
    try:
        with (evidence/f'{phase}.log').open('w') as log:
            worker=subprocess.Popen([sys.executable,'-m','pipeline.worker','--phase',phase],cwd=root,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=pdeath)
            (evidence/f'{phase}_pids.json').write_text(json.dumps({'guard':guard_pid,'worker':worker.pid}))
            while worker.poll() is None:
                if flag[0]:reason='external_signal';break
                if time.monotonic()-start>=cap-10:reason='fixed_budget_cap';break
                time.sleep(.25)
            if worker.poll() is None:
                os.killpg(worker.pid,signal.SIGTERM)
                try:worker.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
            code=worker.returncode
    finally:
        end={'event':'end','phase':phase,'epoch':time.time(),'exit_code':code,'reason':reason,
             'gpu_device_seconds':time.monotonic()-start,'cap_seconds':cap,'cloud_calls':0,'automatic_retries':0,
             'scope':'all assigned load/cache/diagnostics/fit/generation/saving/failure/idle; only owned process group'}
        append(end);(evidence/f'{phase}_completion.json').write_text(json.dumps(end,indent=2))
    return end

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['diagnostics','cache',*C.METHODS]);args=parser.parse_args()
    print(json.dumps(run_phase(args.phase),indent=2))
if __name__=='__main__':main()
