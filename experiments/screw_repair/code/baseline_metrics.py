import csv,json
import numpy as np
from common import ROOT,config,dump,PAIRS,sha

def metrics(pred,y,cfg):
    pred=pred.reshape(-1,4,3,3);y=y.reshape(pred.shape);limit=cfg['decision']['opening_limit_mm']
    approve=pred[:,:,0,:]<=limit;truth=y[:,:,0,:]<=limit
    costs=np.asarray(cfg['decision']['repair_cost']);penalty=cfg['decision']['unsafe_penalty']
    si=cfg['forces_N'].index(cfg['decision']['service_force_N'])
    guessed=(costs+penalty*(pred[:,:,:,si]>limit)).argmin(-1)
    truecost=costs+penalty*(y[:,:,:,si]>limit);optimal=truecost.argmin(-1)
    actual=np.take_along_axis(y[:,:,:,si],guessed[...,None],-1)[...,0]
    selectedcost=np.take_along_axis(truecost,guessed[...,None],-1)[...,0]
    regret=selectedcost-truecost.min(-1)
    robust=np.abs(y[:,:,0,:]-limit)>.04
    unsafe_robust=robust&~truth
    per={'MAE_mm':np.abs(pred-y).mean((1,2,3)),
         'false_approval_num':(approve&~truth).sum((1,2)),'unsafe_cases':(~truth).sum((1,2)),
         'robust_false_approval_num':(approve&unsafe_robust).sum((1,2)),'robust_unsafe_cases':unsafe_robust.sum((1,2)),
         'approval_error_num':(approve!=truth).sum((1,2)),'approval_cases':np.full(len(y),12),
         'repair_regret':regret.mean(1),'selected_repair_unsafe':(actual>limit).mean(1),
         'repair_action_error':(guessed!=optimal).mean(1)}
    eq_n=np.zeros(len(y));eq_d=eq_n.copy();dp=eq_n.copy();de=eq_n.copy();qbad=eq_n.copy();qall=eq_n.copy();prot=eq_n.copy();wipe=eq_n.copy()
    for a,b in PAIRS:
        tg=np.max(np.abs(y[:,a]-y[:,b]),(1,2));pg=np.max(np.abs(pred[:,a]-pred[:,b]),(1,2))
        eq=tg<1e-6;distinct=tg>=.08
        eq_n+=eq&(pg>.04);eq_d+=eq
        mismatch=np.any(approve[:,a]!=approve[:,b],1)|(guessed[:,a]!=guessed[:,b])
        de+=eq&mismatch;dp+=eq
        # Same physical targets, changed decisions, with measured error/regret.
        harmed=(np.any((approve[:,a]!=truth[:,a])|(approve[:,b]!=truth[:,b]),1)|
                (regret[:,a]>=.1)|(regret[:,b]>=.1))
        qbad+=eq&mismatch&harmed;qall+=eq
        prot+=distinct;wipe+=distinct&(pg<=.04)
    per.update(eq_gap_num=eq_n,eq_pairs=eq_d,eq_decision_num=de,eq_decision_pairs=dp,
        eq_harm_num=qbad,eq_harm_pairs=qall,protected_false_merge_num=wipe,protected_pairs=prot)
    def ratio(a,b):return float(per[a].sum()/max(per[b].sum(),1))
    row={'MAE_mm':float(per['MAE_mm'].mean()),'false_approval':ratio('false_approval_num','unsafe_cases'),
        'robust_false_approval':ratio('robust_false_approval_num','robust_unsafe_cases'),
        'approval_error':ratio('approval_error_num','approval_cases'),'repair_regret':float(regret.mean()),
        'selected_repair_unsafe':float((actual>limit).mean()),'repair_action_error':float((guessed!=optimal).mean()),
        'equivalent_disagreement':ratio('eq_gap_num','eq_pairs'),'equivalent_decision_disagreement':ratio('eq_decision_num','eq_decision_pairs'),
        'equivalent_decision_harm':ratio('eq_harm_num','eq_harm_pairs'),
        'protected_false_merge':ratio('protected_false_merge_num','protected_pairs'),
        'unsafe_cases':int(per['unsafe_cases'].sum()),'robust_unsafe_cases':int(per['robust_unsafe_cases'].sum()),
        'equivalent_pairs':int(eq_d.sum()),'protected_pairs':int(prot.sum())}
    return row,per

def pass_cell(r):
    return (r['MAE_mm']<=.05 and r['false_approval']<=.05 and r['equivalent_disagreement']<=.10
        and r['equivalent_decision_disagreement']<=.05 and r['repair_regret']<=.10 and r['protected_false_merge']<=.10)

def analyze():
    cfg=config();rows=[];selected=json.loads((ROOT/'evidence/FIT_FREEZE.json').read_text())['selected_checkpoints']
    for item in selected:
      for split in cfg['test_splits']:
        source=ROOT/'predictions'/f'draw{item["draw"]}'/f'{item["arm"]}_{item["objective"]}_s{item["seed"]}_b{item["budget"]}_{split}.npz'
        with np.load(source) as d:r,per=metrics(d['predictions'],d['targets'],cfg)
        r.update({k:item[k] for k in ['draw','arm','objective','seed','budget','selected_lr','selected_epoch','validation_plateau_400_to_1200']})
        r.update(split=split,passes_gate=pass_cell(r));rows.append(r)
        np.savez_compressed(source.with_name(source.stem+'_per_family.npz'),**per)
    ordinary_solutions=[]
    for arm in cfg['arms']:
      for objective in cfg['objectives']:
        group=[r for r in rows if r['arm']==arm and r['objective']==objective and r['budget']==max(cfg['budgets']) and r['split'] in cfg['heldout_operations']]
        if all(r['passes_gate'] for r in group):ordinary_solutions.append({'arm':arm,'objective':objective,'cells':len(group)})
    visual=[x for x in ordinary_solutions if cfg['arms'][x['arm']]['encoder']!='decoded']
    numeric=[x for x in ordinary_solutions if cfg['arms'][x['arm']]['encoder']=='decoded']
    strong=[]
    for arm in ('resnet_transformer','siglip_transformer'):
        group=[r for r in rows if r['arm']==arm and r['objective']=='paired' and r['budget']==max(cfg['budgets'])]
        seen=[r for r in group if r['split']=='seen_repair'];held=[r for r in group if r['split'] in cfg['heldout_operations']]
        strong.append(all(r['MAE_mm']<=.05 and r['false_approval']<=.05 and r['validation_plateau_400_to_1200'] for r in seen)
            and all(r['equivalent_decision_harm']>.05 and r['robust_false_approval']>.10 for r in held))
    if visual: decision='STOP_SPECIAL_HISTORY_ROUTE_ORDINARY_VISUAL_REMEDY_PASSES'
    elif numeric:decision='OBSERVATION_OR_REPRESENTATION_LIMIT_NOT_HISTORY_NECESSITY'
    elif all(strong):decision='CONTINUE_CONTROLLED_DIAGNOSIS_ONLY_NOT_METHOD_OR_NOVELTY_PROOF'
    else:decision='INCONCLUSIVE_DO_NOT_EXPAND'
    dest=ROOT/'analysis';dest.mkdir(exist_ok=True)
    with (dest/'all_cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(dest/'SUMMARY.json',{'decision':decision,'ordinary_solutions':ordinary_solutions,'cells':rows,
        'scope':'instrumented simulation; frozen ResNet18/SigLIP2 + small temporal heads, not modern world-model or real hardware',
        'causal_scope':'matched equivalent histories demonstrate output/decision dependence; does not identify latent-memory mechanism',
        'checkpoint_freeze_sha256':sha(ROOT/'evidence/FIT_FREEZE.json')})
    lines=['# 未见修复操作验证：212 实际结果','',f'预注册判定：**{decision}**。','',
        '表中为两套互不重叠数据、三个初始化种子的描述均值；通过规则检查每个单元，测试操作使用匹配家族。',
        '','|模型|监督|测试操作|MAE mm|误放行|等价预测分歧|等价决策分歧|维修代价差|通过单元|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in cfg['arms']:
     for obj in cfg['objectives']:
      for split in cfg['test_splits']:
        g=[r for r in rows if r['arm']==arm and r['objective']==obj and r['split']==split and r['budget']==max(cfg['budgets'])]
        avg=lambda k:np.mean([r[k] for r in g])
        lines.append(f'|{arm}|{obj}|{split}|{avg("MAE_mm"):.5f}|{avg("false_approval"):.2%}|{avg("equivalent_disagreement"):.2%}|{avg("equivalent_decision_disagreement"):.2%}|{avg("repair_regret"):.4f}|{sum(r["passes_gate"] for r in g)}/{len(g)}|')
    lines+=['','预算为400/1200轮的嵌套前缀，两个固定学习率都充分执行后按验证响应误差选取。最多1200轮不构成全局收敛证明；平台期单独记录。',
        '','所有承重和维修决策由冻结预测生成，然后在同一声明物理规律的副本上执行。输出分歧、决策变化与损害分别统计；不把分歧率直接当作事故率。',
        '','局部补强将损伤变量减少55%；支撑增加240 N/mm并保留损伤。二者操作在训练/验证中完全不存在，但训练覆盖已有支撑0/120/240/360/480及损伤范围，测试新支撑240/360/480仍在范围内。每段历史同样含12帧与两次非破坏性修复后测量，未把操作身份或私有状态传入模型。',
        '','模型使用公开仪表仿真图像。SigLIP2是冻结现代视觉编码器，不是视频世界模型；预训练数据和成本无法跨编码器匹配。匹配的是本任务的数据、更新次数、学习率搜索和验证选择规则。',
        '','测试阈值和维修成本属于本仿真协议，不是实物安全标准。新修复物理规律未经过硬件标定；像素参照只证明本设置可识别。']
    (ROOT/'RESULTS_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('DECISION',decision,flush=True)
    return decision

if __name__=='__main__':analyze()
