"""Ideal cohesive-interface experiment, not calibrated screw hardware.

Repair operations alter preload, cohesive memory or parallel elastic support.
Observed displacement-controlled probes are non-destructive. Future force tests
are independent interventions on fresh copies, never observed during inference.
"""
import numpy as np
from physics_prior import damage

def virtual_peak(d):
    d=np.asarray(d,float)
    return np.where(d<=0, .12, .24/(1.2-d))

def repair(k,p,peak,brace,operation):
    k,p,peak,brace=np.broadcast_arrays(k,p,peak,brace)
    if operation=='retighten': return k,np.full_like(p,160),peak,brace
    if operation=='replace_interface': return k,np.full_like(p,160),np.full_like(peak,.12),brace
    if operation=='partial_patch': return k,np.full_like(p,160),virtual_peak(damage(peak)*.45),brace
    if operation=='brace_install': return k,np.full_like(p,160),peak,brace+240.
    raise ValueError(operation)

def rollout(k,p,peak,brace,forces=(230,270,310),dt=.001):
    k,p,peak,brace=np.broadcast_arrays(k,p,peak,brace)
    shape=k.shape+(len(forces),)
    k=k[...,None]; p=p[...,None]; brace=brace[...,None]
    kap=np.broadcast_to(peak[...,None],shape).copy()
    x=np.zeros(shape,float)
    for step in range(1,round(2/dt)+1):
        force=np.asarray(forces)*min(step*dt/.2,1.)
        reaction=((1-damage(kap))*k+brace)*x
        x=np.clip(x+dt*(np.maximum(force-p,0)-reaction)/10,0,1.6)
        kap=np.maximum(kap,x)
    return x

def outcomes(states,dt=.001):
    k,p,peak,brace=np.moveaxis(np.asarray(states),-1,0)
    result=[]
    for act in ('keep','retighten_220','replace_interface'):
        pp=p if act=='keep' else np.full_like(p,220 if act=='retighten_220' else 160)
        pk=peak if act!='replace_interface' else np.full_like(peak,.12)
        result.append(rollout(k,pp,pk,brace,dt=dt))
    return np.stack(result,-2)

def reference_from_visible(decoded):
    # Channel values are measurements decoded from the public seven-segment RGB.
    # Calibration frame: unloaded intact interface, imposed x=.1, F=k*x.
    k=decoded[...,0,0]/decoded[...,0,2]-decoded[...,0,3]
    p=decoded[...,-1,1]; brace=decoded[...,-1,3]
    tangent=(decoded[...,-2,0]-decoded[...,-2,1])/decoded[...,-2,2]
    d=np.clip(1-(tangent-brace)/k,0,1)
    states=np.stack([k,p,virtual_peak(d),brace],-1)
    return outcomes(states),states

def decisions(pred,cfg):
    pred=np.asarray(pred).reshape(*np.asarray(pred).shape[:-1],3,3)
    unsafe=pred>cfg['decision']['opening_limit_mm']
    si=cfg['forces_N'].index(cfg['decision']['service_force_N'])
    score=np.asarray(cfg['decision']['repair_cost'])+cfg['decision']['unsafe_penalty']*unsafe[...,si]
    selected=score.argmin(-1)  # deterministic tie break: keep, retighten, replace
    return ~unsafe[...,0,:],selected

def verify():
    s=np.array([[900,160,.12,0],[900,160,.6,0]],float)
    y=outcomes(s)
    assert y[1,0,1]>y[0,0,1]
    new=repair(*s.T,'partial_patch')
    assert 0<damage(new[2][1])<damage(s[1,2])
    br=np.stack(repair(*s.T,'brace_install'),-1)
    assert (outcomes(br)[:,0]<=y[:,0]+1e-12).all()
    rep=np.stack(repair(*s.T,'replace_interface'),-1)
    assert np.allclose(outcomes(rep)[0],outcomes(rep)[1])
    halved=outcomes(s,dt=.0005)
    gap=float(np.abs(halved-y).max()); assert gap<.02
    return {'damage_preserved':True,'partial_damage_reduction':True,'brace_reduces_opening':True,
            'replacement_erases_damage':True,'dt_halving_max_mm':gap}
