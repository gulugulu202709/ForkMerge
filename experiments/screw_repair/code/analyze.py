import csv,json
import numpy as np
from common import ROOT,config,dump,sha,PAIRS
from baseline_metrics import metrics,pass_cell
from physics import decisions,rollout

def analyze():
    cfg=config();items=json.loads((ROOT/'evidence/FIT_FREEZE.json').read_text())['selected_checkpoints']
    rows=[];family={};dest=ROOT/'analysis';dest.mkdir(exist_ok=True);physical={}
    for draw in cfg['draws']:
      for split in cfg['test_splits']:
        with np.load(ROOT/'data'/f'draw{draw}'/(split+'_audit.npz')) as d:
            physical[draw,split]={k:d[k] for k in ('states','decoded','family_ids')}
    for item in items:
      for split in cfg['test_splits']:
        path=ROOT/'predictions'/f'draw{item["draw"]}'/f'{item["arm"]}_s{item["seed"]}_{split}.npz'
        with np.load(path) as d:pred=d['predictions'];y=d['targets'];ids=d['family_ids']
        audit=physical[item['draw'],split];assert np.array_equal(ids,audit['family_ids'])
        row,per=metrics(pred,y,cfg);approve,chosen=decisions(pred,cfg)
        st=audit['states'];k,p,pk,b=np.moveaxis(st,-1,0)
        pp=np.where(chosen==0,p,np.where(chosen==1,220.,160.));peak=np.where(chosen==2,.12,pk)
        executed=rollout(k,pp,peak,b,forces=(cfg['decision']['service_force_N'],))[...,0]
        yy=y.reshape(-1,4,3,3);si=cfg['forces_N'].index(cfg['decision']['service_force_N']);limit=cfg['decision']['opening_limit_mm']
        lookup=np.take_along_axis(yy[:,:,:,si],chosen[...,None],-1)[...,0]
        assert np.max(np.abs(executed-lookup))<1e-6
        assert np.array_equal(executed>limit,lookup>limit)
        unsafe=executed>limit;true_safe=yy[:,:,0,:]<=limit
        costs=np.asarray(cfg['decision']['repair_cost']);penalty=cfg['decision']['unsafe_penalty']
        truecost=costs+penalty*(yy[:,:,:,si]>limit);actualcost=costs[chosen]+penalty*unsafe
        regret=actualcost-truecost.min(-1)
        strict=np.zeros(len(ids),int);hazard=strict.copy();bearing_hazard=strict.copy();repair_hazard=strict.copy();gaps=strict.copy()
        pred9=pred.reshape(-1,4,9)
        for a,bidx in PAIRS:
            eq=np.max(np.abs(st[:,a]-st[:,bidx]),1)<1e-8
            posteq=np.max(np.abs(audit['decoded'][:,a,6:]-audit['decoded'][:,bidx,6:]),(1,2))<1e-10
            assert not (eq&~posteq).any()
            bh=np.any((approve[:,a]!=approve[:,bidx])&~true_safe[:,a],1)
            rh=(chosen[:,a]!=chosen[:,bidx])&(unsafe[:,a]!=unsafe[:,bidx])
            strict+=eq;bearing_hazard+=eq&bh;repair_hazard+=eq&rh;hazard+=eq&(bh|rh)
            gaps+=eq&(np.max(np.abs(pred9[:,a]-pred9[:,bidx]),1)>.04)
        row.update({k:item[k] for k in ('draw','arm','seed','selected_lr','selected_epoch','origin')})
        row.update(split=split,repair_regret=float(regret.mean()),selected_repair_unsafe=float(unsafe.mean()),
           passes_gate=pass_cell(row),same_state_pairs=int(strict.sum()),history_unsafe_choice_pairs=int(hazard.sum()),
           history_unsafe_choice_families=int((hazard>0).sum()),bearing_unsafe_change_pairs=int(bearing_hazard.sum()),
           maintenance_safety_disagreement_pairs=int(repair_hazard.sum()),unsafe_keep_families=int((~true_safe).any((1,2)).sum()),
           maintenance_rollout_vs_lookup_max_mm=float(np.max(np.abs(executed-lookup))))
        row['passes_gate']=pass_cell(row)
        arrays={**per,'family_ids':ids,'repair_regret':regret.mean(1),'maintenance_unsafe_num':unsafe.sum(1),
          'maintenance_cases':np.full(len(ids),4),'same_state_pairs':strict,'history_hazard_pairs':hazard,
          'bearing_hazard_pairs':bearing_hazard,'maintenance_hazard_pairs':repair_hazard,'prediction_gap_pairs':gaps,
          'chosen_repair':chosen,'executed_opening_mm':executed,'realized_branch_regret':regret}
        pf=dest/'per_family'/f'draw{item["draw"]}_{item["arm"]}_s{item["seed"]}_{split}.npz';pf.parent.mkdir(exist_ok=True)
        np.savez_compressed(pf,**arrays);family[item['arm'],split,item['draw'],item['seed']]=arrays;rows.append(row)
    with (dest/'all_cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gate=cfg['pilot_continue_gate'];rng=np.random.default_rng(gate['bootstrap_seed']);n=cfg['confirmation_families_per_draw']
    boot={d:rng.integers(0,n,size=(gate['bootstrap_replicates'],n)) for d in cfg['draws']};contrasts=[]
    for control in [*cfg['primary_controls'],'tf_plain','decoded_pair']:
      for split in cfg['heldout_operations']:
        gm=[r for r in rows if r['arm']=='canonical_scope' and r['split']==split]
        gc=[r for r in rows if r['arm']==control and r['split']==split]
        pairs={(r['draw'],r['seed']):r for r in gc};gc=[pairs[r['draw'],r['seed']] for r in gm]
        m=float(np.mean([r['repair_regret'] for r in gm]));c=float(np.mean([r['repair_regret'] for r in gc]))
        deltas=[];indices=[]
        for d in cfg['draws']:
            a=np.mean([family['canonical_scope',split,d,s]['repair_regret'] for s in cfg['seeds']],0)
            b=np.mean([family[control,split,d,s]['repair_regret'] for s in cfg['seeds']],0)
            deltas.append(a-b);indices.append((a-b)[boot[d]].mean(1))
        upper=float(np.quantile(np.mean(indices,0),gate['bootstrap_upper_quantile']))
        fa=float(np.mean([a['false_approval']-b['false_approval'] for a,b in zip(gm,gc)]))
        mu=float(np.mean([a['selected_repair_unsafe']-b['selected_repair_unsafe'] for a,b in zip(gm,gc)]))
        nonworse=sum(a['repair_regret']<=b['repair_regret']+1e-12 for a,b in zip(gm,gc))
        fabad=sum(a['false_approval']-b['false_approval']>gate['noninferiority_false_approval_margin'] for a,b in zip(gm,gc))
        mubad=sum(a['selected_repair_unsafe']-b['selected_repair_unsafe']>gate['noninferiority_maintenance_unsafe_margin'] for a,b in zip(gm,gc))
        reduction=(c-m)/c if c>0 else None
        passed=(reduction is not None and reduction>=gate['regret_relative_reduction_min'] and
            nonworse>=gate['nonworse_paired_cells_per_operation'] and upper<0 and
            fa<=gate['noninferiority_false_approval_margin'] and mu<=gate['noninferiority_maintenance_unsafe_margin'] and fabad<=1 and mubad<=1)
        contrasts.append({'control':control,'split':split,'method_regret':m,'control_regret':c,
          'regret_relative_reduction':reduction,'regret_difference':m-c,'family_bootstrap_upper':upper,
          'false_approval_difference':fa,'maintenance_unsafe_difference':mu,'nonworse_regret_cells':nonworse,
          'safety_margin_exceeded_cells':{'false_approval':fabad,'maintenance_unsafe':mubad},'passes_pilot_gate':bool(passed),
          'primary':control in cfg['primary_controls'],'family_difference_sha256':__import__('hashlib').sha256(np.stack(deltas).tobytes()).hexdigest()})
    primary=[r for r in contrasts if r['primary']]
    decision='CONTINUE_MATCHED_VISUAL_BENEFIT_PILOT_ONLY' if all(r['passes_pilot_gate'] for r in primary) else 'STOP_NO_EXTRA_BENEFIT_IN_THIS_PILOT'
    population_ok=all(all(r[k]>0 for k in ('unsafe_cases','robust_unsafe_cases','equivalent_pairs','protected_pairs'))
        for r in rows if r['split'] in cfg['heldout_operations'])
    if not population_ok:decision='INCONCLUSIVE_EMPTY_POPULATION'
    dump(dest/'SUMMARY.json',{'decision':decision,'evaluation_type':'simulation_only','cells':rows,'contrasts':contrasts,
      'all_heldout_populations_nonempty':population_ok,
      'gate_scope':'4 planned matched-visual regret contrasts, descriptive pilot with family-cluster bootstrap; not population safety or novelty proof',
      'numeric_reference_gate_passed':all(r['passes_pilot_gate'] for r in contrasts if r['control']=='decoded_pair'),
      'checkpoint_freeze_sha256':sha(ROOT/'evidence/FIT_FREEZE.json'),'protocol_sha256':sha(ROOT/'configs/protocol.json')})
    lines=['# 规范状态＋受保护历史证据：直接对照实际结果','',f'冻结判据输出：**{decision}**。完整性和主张审阅另行记录。','',
      '每格为两个新测试draw、三个模型种子的描述均值；相同家族/载荷/种子相关，不能当独立事故。',
      '','|方法|测试操作|MAE mm|危险误放行|维修执行超阈值|真实维修regret|等价决策分歧|原任务过关单元|',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in cfg['arms']:
      for split in cfg['test_splits']:
        g=[r for r in rows if r['arm']==arm and r['split']==split];avg=lambda k:float(np.mean([r[k] for r in g]))
        lines.append(f'|{arm}|{split}|{avg("MAE_mm"):.5f}|{avg("false_approval"):.2%}|{avg("selected_repair_unsafe"):.2%}|{avg("repair_regret"):.5f}|{avg("equivalent_decision_disagreement"):.2%}|{sum(r["passes_gate"] for r in g)}/6|')
    lines+=['','|对照|操作|维修regret相对下降|差值bootstrap上界|非劣regret单元|额外收益筛查|',
       '|---|---|---:|---:|---:|---|']
    for c in contrasts:
        rel='NA' if c['regret_relative_reduction'] is None else f'{c["regret_relative_reduction"]:.2%}'
        lines.append(f'|{c["control"]}|{c["split"]}|{rel}|{c["family_bootstrap_upper"]:.5f}|{c["nonworse_regret_cells"]}/6|{c["passes_pilot_gate"]}|')
    lines+=['','本轮是论文canonical+witness结构的连续响应适配，未增加私有状态或pre-action标签，省略所有方法的pre-action辅助损失。普通scope对照使用同一erase/protect损失以区分结构和目标重加权。',
      '','视觉方法共用同一冻结SigLIP2特征。原plain/paired以及公开RGB读数GRU复用已审计冻结训练，新增24条完整轨迹；全部同数据、更新、2LR搜索与验证选择。不同结构容量/FLOPs、数字读数与视觉表征不声称相同。',
      '','使用1024个全新基础家族，排除前轮所有基础元组；操作面板共享家族且等价终态可重复。旧测试属于开发证据，没有用旧模型测试预测改本轮协议。',
      '','实际选择在私有真实状态副本上重新积分执行。0.35mm阈值、固定成本和无拒绝执行选项属于仿真utility。Bootstrap先平均技术种子再按家族分draw重采样；主要4个regret对比使用98.75%单侧分位上界，只是有限仿真筛查。',
      '','没有真实螺丝、硬件标定、自然视频、主动选测或基础世界模型结果。本轮负面结果仅约束这一固定实现/任务/预算。']
    (ROOT/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('DIRECT_DECISION',decision,flush=True);return decision
if __name__=='__main__':analyze()
