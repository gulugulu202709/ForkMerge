import concurrent.futures,datetime,fcntl,json,os,subprocess,sys,threading,time,traceback
from pathlib import Path
from common import ROOT,config,dump,sha,code_hashes,PYTHON

START=time.time();LOCK=threading.Lock();cfg=config();DEADLINE=START+cfg['gpu_wall_limit_s']
def status(stage,**extra):
    with LOCK:
        row={'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor_pid':os.getpid(),**extra}
        dump(ROOT/'evidence/STATUS.json',row);print('DIRECT_STATUS',json.dumps(row),flush=True)

def command(script,args=(),gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    name=Path(script).stem+'_'+('_'.join(map(str,args)).replace('/','_') or 'default')
    log=ROOT/'evidence'/f'{name}.log';cmd=[PYTHON,str(ROOT/'code'/script),*map(str,args)]
    remaining=DEADLINE-time.time()
    if remaining<=0:raise TimeoutError('Registered campaign wall cap reached')
    with log.open('w') as f:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
    assert result.returncode==0,(cmd,result.returncode,str(log))
    return log

def admit():
    held=[];lockdir=ROOT.parent.parent/'.resource_locks';lockdir.mkdir(exist_ok=True)
    while not held and time.time()<DEADLINE:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        for line in out.splitlines():
            gpu,used=map(int,line.split(','))
            if used>=500:continue
            fd=(lockdir/f'gpu{gpu}.lock').open('a+')
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:fd.close();continue
            again=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if again>=500:fd.close();continue
            held.append((gpu,fd,again))
            if len(held)==cfg['max_gpus']:break
        if not held:status('WAITING_FREE_GPU');time.sleep(30)
    if not held:raise TimeoutError('No free GPU within wall cap')
    dump(ROOT/'evidence/GPU_ADMISSION.json',{'cards':[{'index':g,'used_MiB':m} for g,_,m in held],'threshold_MiB':500})
    return held

def main():
    supervisor=(ROOT/'evidence/supervisor.lock').open('a+');fcntl.flock(supervisor,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=json.loads((ROOT/'evidence/PRELAUNCH_FREEZE.json').read_text())
    assert pre['protocol_sha256']==sha(ROOT/'configs/protocol.json') and pre['code_hashes']==code_hashes()
    assert 'DIRECT_WITNESS_PASS' in (ROOT/'evidence/DOC_WITNESS.stdout.log').read_text()
    meta=json.loads((ROOT/'evidence/DOC_WITNESS.meta.json').read_text());assert meta['exit_code']==0
    assert not (ROOT/'predictions').exists() and not (ROOT/'results').exists()
    status('BUILDING_FRESH_CONFIRMATION');command('build_confirmation.py');command('decode_confirmation.py')
    held=admit();gpus=[g for g,_,_ in held]
    try:
        status('CUDA_WITNESS',gpus=gpus);log=command('witness.py',['--device','cuda'],gpus[0])
        assert 'DIRECT_WITNESS_PASS' in log.read_text()
        status('ENCODING_CONFIRMATION',gpus=gpus)
        def encode_worker(gpu,draws):
            for d in draws:command('extract_confirmation.py',['--draw',d,'--device','cuda'],gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(encode_worker,g,cfg['draws'][i::len(gpus)]) for i,g in enumerate(gpus)]
            for f in futures:f.result()
        tasks=[(d,a,s,lr) for d in cfg['draws'] for a in ('tf_scope','canonical_scope') for s in cfg['seeds'] for lr in cfg['learning_rates']]
        iterator=iter(tasks);tasklock=threading.Lock();counter=0
        def worker(gpu):
            nonlocal counter
            while True:
                with tasklock:task=next(iterator,None)
                if task is None:return
                d,a,s,lr=task
                command('train.py',['--draw',d,'--arm',a,'--seed',s,'--lr',lr,'--device','cuda'],gpu)
                path=ROOT/'results'/f'draw{d}'/f'{a}_s{s}_lr{lr:g}';info=json.loads((path/'FIT.json').read_text())
                assert info['epochs']==cfg['epochs'] and info['steps']==cfg['updates_per_fit']
                assert sha(path/'best.pt')==info['checkpoint_sha256']
                with tasklock:counter+=1;count=counter
                status('TRAINING',new_fits_complete=count,new_fits_expected=len(tasks),reused_fits=36,gpus=gpus,last_completed=task)
        status('TRAINING',new_fits_complete=0,new_fits_expected=24,reused_fits=36,gpus=gpus)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,g) for g in gpus]
            for f in futures:f.result()
        assert counter==24
        assert pre['protocol_sha256']==sha(ROOT/'configs/protocol.json') and pre['code_hashes']==code_hashes()
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpus[0]),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2')
        import evaluate,analyze
        status('FREEZING_ALL_FITS',new_fits_complete=24,gpus=gpus);evaluate.freeze()
        status('EVALUATING_CONFIRMATION',new_fits_complete=24,gpus=gpus);evaluate.evaluate('cuda')
        status('ANALYZING_REALIZED_DECISIONS',new_fits_complete=24,gpus=gpus);decision=analyze.analyze()
        status('COMPLETE_PENDING_AUDIT',new_fits_complete=24,reused_fits=36,selected_checkpoints=30,predictions=90,
            decision=decision,gpus=gpus,seconds=time.time()-START)
    finally:
        for _,fd,_ in held:fd.close()

if __name__=='__main__':
    try:main()
    except Exception as e:status('FAILED',error=str(e));traceback.print_exc();raise
