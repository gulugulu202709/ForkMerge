"""CPU report from real saved artifacts only. No model/tensor fabrication."""
import html,json
from pathlib import Path
from .config import ROOT,PRIOR_GPU_SECONDS,TOTAL_GPU_SECONDS
from .policy import read_json,phase_gate
from .costs import cost_summary

def render(root=ROOT):
    root=Path(root);state=root/'pipeline_state';state.mkdir(exist_ok=True)
    source=read_json(root/'pipeline_data/SOURCE_READY.json',{})
    calibration=read_json(state/'measurement.json',{})
    costs=cost_summary(root)
    progress=[json.loads(x) for x in (state/'progress.jsonl').read_text().splitlines() if x.strip()] if (state/'progress.jsonl').exists() else []
    gates={p:phase_gate(root,p) for p in ('diagnostics','cache','paired','paired_worst_witness')}
    method=[];videos=[];memory=[]
    for arm,title in [('paired','普通更新器＋现有配对监督'),('paired_worst_witness','普通更新器＋候选新监督')]:
        complete=read_json(root/'pipeline_results'/arm/'COMPLETE.json',{})
        score=read_json(state/'reviews'/arm/'score.json',{})
        updates=sum(p.get('stage')=='TRAIN_UPDATE' and p.get('arm')==arm for p in progress)
        actual_samples=[p for p in progress if p.get('stage')=='NATIVE_FORECAST_SAVED' and p.get('arm')==arm]
        method.append(f'<tr><td>{title}</td><td>{updates}</td><td>{len(actual_samples)}</td><td>{"等待实际结果或复核" if not score.get("complete") else "可读性不足／未决" if not score.get("judgeable") else "本次样本全对" if score.get("all_correct") else "存在经复核的错误"}</td></tr>')
        for sample in actual_samples:
            path=Path(sample['video']);rel=path.relative_to(root)
            videos.append(f'<article><h3>{title} · {html.escape(sample["clip_id"])}</h3><video controls src="../{html.escape(str(rel).replace(chr(92),"/"))}"></video><p>实际保存的模型输出；未复核可见性不能据此评价状态对错。</p></article>')
            info=path.with_suffix('.memory.json')
            if info.exists():
                m=read_json(info)
                memory.append({'clip_id':sample['clip_id'],'arm':arm,'seed':sample['seed'],**m})
    diag=[]
    completed=read_json(root/'pipeline_results/diagnostics/COMPLETE.json',{})
    for sample in completed.get('samples',[]):
        rel=Path(sample['video']).relative_to(root)
        diag.append(f'<article><strong>真实未来终点条件下的渲染诊断：{html.escape(sample["scenario"])}</strong><video controls src="../{html.escape(str(rel).replace(chr(92),"/"))}"></video><p>此处使用了真实终点，不是方法预测或历史记忆效果。</p></article>')
    (state/'actual_memory_statistics.json').write_text(json.dumps({'available':bool(memory),'rows':memory,'meaning':'Actual native-fit tensor statistics only; no semantic erase/protect inference.'},indent=2))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三个日常案例：代码准备与真实结果入口</title><style>body{max-width:1120px;margin:35px auto;padding:0 24px 40px;font:17px/1.7 "Microsoft YaHei",system-ui;background:#f4f6fa;color:#1b2b3f}h1{font-size:30px}h2{font-size:23px;margin-top:30px}.note{background:#fff3dd;padding:18px;border-left:4px solid #bd8423}table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #dce3ed;text-align:left}article{background:white;padding:18px;margin:16px 0}video{width:100%}a{color:#2b609a}pre{font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}</style>'''
    page+='<h1>接入与对照代码已准备；真实检查仍等待视频</h1><div class="note">CPU接口检查与真实实验分开记录。未收到的录像、未运行的模型和未完成的复核不会显示为成功结果。9条校准片排除在正式测试之外。</div>'
    page+='<p><a href="../recording/pipeline_annotations_template.json">9条校准录制清单</a> · <a href="../recording/pipeline_annotations_template.json">录制记录表</a> · <a href="../docs/OPERATOR_ZH.md">接入与执行说明</a> · <a href="../README.md">CPU检查记录入口</a></p>'
    page+='<h2>检查与停止规则</h2><p>原始视频检查→实际读数校准→三个生成器诊断→人工复核可见性和动作→冻结正式数据与共享缓存→普通监督→复核→普通方法全对则停止；达到预设有限训练检查后仍有可读错误才执行候选新监督。生成与测量未通过、训练不足或预算耗尽，均记为未决。</p>'
    page+='<p>复核包隐藏单条方法标签与正确答案，但执行阶段和顺序可能被操作者知道；不声称已做独立双盲。读出类型与位置会明确提示，实际答案留空。</p>'
    page+='<table><tr><th>项目</th><th>当前状态</th></tr>'+f'<tr><td>源视频接入</td><td>{"已建立校准数据" if source.get("calibration_ready") else "等待实际视频及观察标注"}</td></tr><tr><td>测量校准</td><td>{html.escape(str(calibration.get("status","未运行")))}</td></tr><tr><td>不重叠正式数据</td><td>{"已冻结/验证" if source.get("formal_ready") else "尚未具备，不能用9条校准片代替"}</td></tr></table>'
    page+='<h2>两组实际结果</h2><table><tr><th>方法</th><th>已记载训练更新</th><th>已完成预测</th><th>结论</th></tr>'+''.join(method)+'</table>'
    page+='<p>两组共用冻结Wan2.1-VACE-1.3B、64×1536记忆、普通更新器、32次更新×2种子和同一采样设置。新组只增加同一未来中的左侧／右侧／全图读数区域的配对流误差权重；这不是三条反事实干预轨迹，也尚无新颖性或收益结论。</p>'
    page+=f'<p>原后来阶段总额{TOTAL_GPU_SECONDS:.0f}GPU秒，之前已用{PRIOR_GPU_SECONDS:.4f}秒。后续所有加载、失败、缓存、训练和生成计入同一账本；每组上限2100秒，诊断／共享缓存各450秒。448×256、21帧、16步在本批输出前固定，模型权重不变。</p>'
    page+='<pre>'+html.escape(json.dumps(costs,ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>实际生成的视频</h2>'+(''.join(diag+videos) or '<p>当前没有这三组新场景的生成视频，不展示示意图代替结果。</p>')
    page+='<h2>实际记忆变化</h2>'+('<p>已保存真实训练模型的逐阶段张量与差分范数；这些数字不证明某一语义已经被擦除或保留。</p>' if memory else '<p>尚未训练，没有实际记忆变化可展示。</p>')
    if memory:page+='<pre>'+html.escape(json.dumps(memory,ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>当前放行检查</h2><pre>'+html.escape(json.dumps(gates,ensure_ascii=False,indent=2))+'</pre><p><a href="../validation/CPU_CHECKS.json">CPU检查结果（不是方法结果）</a> · <a href="../docs/PROTOCOL.md">实施假设与限制</a></p></html>'
    dest=root/'pipeline_report/index.html';dest.parent.mkdir(exist_ok=True);dest.write_text(page,encoding='utf-8')
    return {'report':str(dest),'actual_memory_available':bool(memory),'method_results':bool(videos),'no_model_calls':True}

if __name__=='__main__':print(json.dumps(render(),ensure_ascii=False))
