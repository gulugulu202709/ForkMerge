"""Independent reduced-order reference simulators. No learned model imports.

Joint: overdamped opening resisted by preload and a bilinear cohesive element.
The cohesive element represents an idealized damaged joint interface, NOT a
calibrated screw thread or torque-to-preload model. Replacement resets damage;
retightening changes preload only. Chemistry: two perfectly stagnant zones
until ideal complete mixing; dilute strong acid/base with water ionization.
"""
import numpy as np


def damage(peak, onset=.2, failure=1.2):
    peak = np.asarray(peak, dtype=np.float64)
    return np.clip(failure * (peak-onset) / (np.maximum(peak, onset)*(failure-onset)), 0, 1)


def joint_probe(stiffness, preload, peak, forces=(150, 230, 310), dt=.001):
    k, p, history = np.broadcast_arrays(stiffness, preload, peak)
    shape = k.shape + (len(forces),)
    k, p = k[..., None], p[..., None]
    kap = np.broadcast_to(history[..., None], shape).copy()
    opening = np.zeros(shape, dtype=np.float64)
    times = np.linspace(0, 2, 41)
    snapshots = [opening.copy()]
    damages = [damage(kap)]
    steps = int(round(2/dt)); stride = steps//40
    for step in range(1, steps+1):
        force = np.asarray(forces)*min(step*dt/.2, 1.)
        # Nonnegative opening and a physical travel stop; reaction at stop is
        # intentionally not reported as a calibrated force measurement.
        opening = np.clip(opening + dt*(np.maximum(force-p, 0)-(1-damage(kap))*k*opening)/10., 0, 1.6)
        kap = np.maximum(kap, opening)
        if step % stride == 0:
            snapshots.append(opening.copy()); damages.append(damage(kap))
    return np.stack(snapshots, -1), np.stack(damages, -1), times


def ph(excess_mol, volume_L, kw=1e-14):
    c = np.asarray(excess_mol)/np.asarray(volume_L)
    root = np.sqrt(c*c+4*kw)
    # Stable acid and base branches avoid cancellation in alkaline solutions.
    h = np.where(c >= 0, (c+root)/2, 2*kw/(root-c))
    return -np.log10(h)


def mix_zones(state):
    """state = [acid_top, base_top, volume_top, acid_bottom, base_bottom, volume_bottom]."""
    s = np.asarray(state, dtype=np.float64)
    total = s[..., :3]+s[..., 3:]
    return np.concatenate([total/2, total/2], axis=-1)


def chemical_probe(state):
    s = np.asarray(state)
    return np.stack([ph(s[..., 0]-s[..., 1], s[..., 2]),
                     ph(s[..., 3]-s[..., 4], s[..., 5]),
                     ph(s[..., 0]+s[..., 3]-s[..., 1]-s[..., 4]-1e-5,
                        s[..., 2]+s[..., 5]+.001)], -1)


def indicator_rgb(pH):
    # Illustrative BTB-like equilibrium palette, not a measured spectrum.
    fraction = 1/(1+10.**(7.1-np.asarray(pH)))
    yellow = np.array([242., 207., 52.]); blue = np.array([43., 109., 194.])
    return np.rint(yellow+(blue-yellow)*fraction[..., None]).astype(np.uint8)


def verify():
    checks = {}
    d = damage([0, .1, .2, .3, .6, 1.2, 1.5])
    assert np.all(np.diff(d)>=0) and d[0]==0 and d[-1]==1
    checks['bounded_monotone_damage'] = True
    x, dd, t = joint_probe(np.array([900.,900.]), np.array([160.,160.]), np.array([.1,.6]))
    assert np.all(np.diff(dd, axis=-1)>=-1e-12)
    assert x[1,1,-1] > x[0,1,-1]
    checks['repair_retains_damage_and_changes_future'] = True
    xx, _, _ = joint_probe(np.array([900.,900.]), np.array([160.,160.]), np.array([.1,.6]), dt=.0005)
    checks['dt_halving_max_gap_difference_mm'] = float(np.max(np.abs(xx-x)))
    assert checks['dt_halving_max_gap_difference_mm'] < .02
    zero, _, _ = joint_probe(900., 400., .1)
    assert np.max(zero)==0
    checks['subpreload_no_opening'] = True
    assert abs(float(ph(0, .03))-7)<1e-10
    assert abs(float(ph(3e-5,.03))-3)<1e-6
    assert abs(float(ph(-3e-5,.03))-11)<1e-6
    checks['neutral_acid_base_analytic_limits'] = True
    s=np.array([3e-5,0,.02,0,0,.01]); mixed=mix_zones(s)
    assert np.allclose(s[:3]+s[3:],mixed[:3]+mixed[3:],rtol=0,atol=1e-18)
    assert np.allclose(chemical_probe(mixed)[:2],3,atol=1e-6)
    checks['mixing_conserves_acid_base_inventory_and_volume'] = True
    return checks


if __name__=='__main__':
    import json
    print(json.dumps(verify(), indent=2))
