"""One bounded CPU-only preparation verification, independent of GPU budget."""
import argparse,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    if args.attempt<1:raise ValueError('invalid CPU check attempt')
    out=ROOT/'prep-stage'
    suffix='' if args.attempt==1 else f'_{args.attempt:03d}'
    prior=sum(json.loads(p.read_text())['cpu_seconds'] for p in out.glob('CPU_COMPLETION*.json'))
    cap=min(600-prior,14400-3177-prior)
    if cap<10:raise RuntimeError('CPU preparation budget exhausted')
    with (out/f'CPU_STARTED{suffix}.json').open('x') as f:json.dump({'epoch':time.time(),'cpu_cap_seconds':cap,'gpu_device_seconds':0},f)
    start=time.monotonic()
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(math.ceil(cap),math.ceil(cap)+1))
        available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:2])
    def group_cpu(group):
        ticks=os.sysconf('SC_CLK_TCK');total=0
        for p in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields=p.read_text().rsplit(')',1)[1].split()
                if int(fields[2])==group:total+=sum(int(fields[i]) for i in (11,12,13,14))
            except (OSError,ValueError,IndexError):continue
        return total/ticks
    reason='child_exit'
    with (out/f'CPU_CHECKS{suffix}.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'-m','pipeline.prep_checks'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        while p.poll() is None:
            if group_cpu(p.pid)>=cap-3:reason='aggregate_cpu_cap';break
            if time.monotonic()-start>=900:reason='wall_cap';break
            time.sleep(.2)
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    usage=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=usage.ru_utime+usage.ru_stime
    receipt={'exit_code':p.returncode,'reason':reason,'cpu_seconds':cpu,'wall_seconds':time.monotonic()-start,
             'prior_prep_cpu_seconds':prior,'prep_cpu_cap_seconds':600,
             'prior_and_exploration_reserved_cpu_seconds':3177,'cumulative_reserved_cpu_seconds':3177+prior+cpu,
             'total_cpu_cap_seconds':14400,'gpu_device_seconds':0,'cloud_calls':0,'automatic_retries':0}
    (out/f'CPU_COMPLETION{suffix}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
    if p.returncode:print((out/f'CPU_CHECKS{suffix}.log').read_text()[-10000:])
    return p.returncode
if __name__=='__main__':sys.exit(main())
