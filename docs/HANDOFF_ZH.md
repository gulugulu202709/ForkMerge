# 给接手同学的使用说明

## 第一天先跑什么

按首页只安装 `requirements-replay.txt`，运行 `python tools/reproduce_screw.py`。不需要作者机器、服务器、GPU或模型权重。打开 `runtime/screw-replay/RESULTS_REPORT.md` 与 `VERIFICATION.json`：脚本从实际预测重算原结果，而不是打印已有表格。

再按首页安装CPU版PyTorch与 `requirements-cpu-tested.txt`，运行 `python tools/smoke_cpu.py` 和全部软件测试。读取演示张量：

```python
import numpy as np
data = np.load('runtime/cpu-smoke/paired_memory.npz', allow_pickle=False)
print({key: data[key].shape for key in data.files})
```

`before/after` 是一次CPU参数更新前后的实际张量；`history_stages` 是四阶段中间记忆。没有语义因子真值，不能把范数或差分热图解释为“损伤已保护/旧螺丝已擦除”。

## 两个版本分别是什么

1. **早期规范状态＋历史证据版本**：`experiments/screw_repair/code/models.py:CanonicalWitness`。配套结果是仪表化损伤界面仿真，含普通视觉对照和读数参考。原停止判据未通过；局部收益与相反结果都保留。完整视觉缓存/检查点未发布，当前支持从预测复算，不支持一键原样重训。
2. **最新两组监督原型**：`pipeline/model.py:OrdinaryUpdater/paired_objective`。两组均为普通门控更新器；候选组加最差读数区域关系项。正式设置是64×1536记忆、两个种子、32次更新、同一冻结生成器。合成演示的一次更新不代替正式实验。尚无完成的实拍两组对照。

研究问题仍是：新动作消除了什么历史差异，留下什么不可逆后果。现有代码不证明“新算子必要”；TinyHistory、未见修复组合的真实未来视频对照等尚待完成。先检查普通方法是否已解决问题，再决定是否改结构。

## 从哪里修改

- **记忆结构**：`pipeline/model.py`。保持 `forward(features, actions, trace=False)` 输入输出契约；不要把未来帧、观测真值或样本ID塞入历史输入。
- **监督**：`paired_objective`、`witness_masks`、`pipeline/worker.py:unit_loss`。三个mask是一个录制未来的不同区域读数，不是三次反事实干预。CPU演示的人工切片也不是原生生成器。
- **生成器**：`pipeline/spine.py:Reader/load_pipe` 与 `pipeline/worker.py`。记忆拼到文本嵌入后，梯度需穿过冻结模型回到更新器。上游接口、精度/设备、原生前向与反向仍需真实输入实测；CPU测试不覆盖这些。
- **旧实验**：保持 `experiments/screw_repair/` 归档不变，在新目录开发。修改归档代码会使校验失败；不能更新预期摘要把新输出冒充旧结果。

模块、形状与文件用途见 [CODE_MAP.md](CODE_MAP.md)。

## 接入实际录像

1. 三类案例为：换杯后洒落痕迹、清洗/重装茶壶后接收杯残留、糖盒清空重装后盒外物体。保留连续原片，不根据预期填答案。
2. 根目录建立 `incoming/`，原片放入；按 `recording/pipeline_annotations_template.json` 建立 `incoming/manifest.json`。模板的planned/status、空值和核验项只有实际观察后才能更新，不能批量勾选。
3. 填写四阶段动作、实际时间、历史截止、揭示窗口、左右ROI和读数，遵守 [CONTRACT](../pipeline/CONTRACT.md)。历史采样不得越过 `history_end`；正式测试未来不能作为生成条件。
4. 先9条校准片，运行 `python -m pipeline.entry prepare`、`python -m pipeline.entry calibrate`。正式train/dev/test另需不重叠录制，最低27条；最小数量只是接口准入，不代表论文统计证据充分。
5. 分开计量动作执行、镜头稳定、目标曝光与状态正确。使用真实终点的渲染诊断不算方法预测。看不清保留unknown，不靠脚本预期补真值。

详细命令与复核字段见 [OPERATOR_ZH.md](OPERATOR_ZH.md)。公开包没有实际原片、完成的校准或真实视频方法权重。

## 原生训练还需要什么

`pipeline.guard` 是原任务的预算续跑守护器，总上限5400 GPU设备秒，包含前次193.3471秒，依赖真实旧账本。公开包没有个人运维记录。独立同学的新实验不应继承或伪造这些记录，也不能直接调用worker绕过检查。

独立新训练应先明确数据版本、生成器版本、两组相同的容量/训练/选点预算、设备和总成本上限、校准/曝光标准及停止条件，再实现对应的新运行入口。这个通用入口**尚未完成**；本次没有改变原协议、门槛和预算，也没有启动GPU。

原生阶段要求Linux。Windows可用本次CPU复算、张量演示、状态检查与软件测试。`prep_guard.py` 是原Linux任务的资源收据工具，不是通用安装步骤。

## 下一步检查顺序

先复算90个单元/8项对照，再跑合成CPU演示和软件测试；接着落实实拍数据、观测标注、读数/曝光校准，最后验证原生生成与反向。缺哪项标哪项。普通方法若解决冻结测试就停止额外结构路线；样本少、欠拟合或生成器不展示目标，不能证明方法必要。
