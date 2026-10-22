# ForkMerge：研究交接版

仓库包含**可复算的早期仿真实验**和**待真实视频验证的生成原型**，供继续研究。它不是已经证明有效的完整论文方法。

|你想做什么|从这里开始|实际提供什么|
|---|---|---|
|复算早期螺丝实验|[螺丝实验说明](experiments/screw_repair/README.md)、`tools/reproduce_screw.py`|90份实跑预测、6份仿真状态数组、科学代码、原始结果；约8 MB|
|运行最新记忆原型|`pipeline/model.py`、`tools/smoke_cpu.py`|CPU前向/反向与真实张量保存，明确是合成接口演示|
|接入实拍视频|[接手指南](docs/HANDOFF_ZH.md)、[数据操作说明](docs/OPERATOR_ZH.md)|数据契约、校准与复核代码；原片和观测标注尚缺|
|继续改方法|[代码地图](docs/CODE_MAP.md)、[证据与缺口](docs/EVIDENCE_STATUS.md)|旧规范状态＋历史证据模型与新普通更新器分开保留|

**先读 [docs/HANDOFF_ZH.md](docs/HANDOFF_ZH.md)。** 建议Python 3.12。下列CPU入口不需要212账号、API密钥或模型权重。

## 第一步：复算已有实际输出

在仓库根目录创建并激活虚拟环境：

```bash
python -m venv .venv
```

Linux/macOS：`source .venv/bin/activate`。Windows PowerShell：`.venv\Scripts\Activate.ps1`；也可以直接用 `.venv\Scripts\python.exe` 代替 `python`，无需修改系统执行策略。

```bash
python -m pip install -r requirements-replay.txt
python tools/reproduce_screw.py
```

成功后 `runtime/screw-replay/VERIFICATION.json` 应显示 `VERIFIED_ARCHIVED_SIMULATION_REPLAY`。脚本校验输入SHA-256，重新执行物理决策与统计，核对90个单元和8项对照。

原结论仍为 **`STOP_NO_EXTRA_BENEFIT_IN_THIS_PILOT`**，负面结果完整保留。“实际输出”是实际运行产生的**仿真预测**，不是实拍螺丝数据，也不是视频生成结果。输出目录已存在时拒绝覆盖；可另指定 `--output runtime/screw-replay-02`。

## 第二步：运行最新原型的CPU接口演示

先安装CPU版PyTorch，再安装已测试依赖；不会下载视频模型：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu-tested.txt
python tools/smoke_cpu.py
python -m pipeline.entry status
python -m unittest discover -s pipeline/tests -v
```

`runtime/cpu-smoke/SMOKE.json` 和两个 `*_memory.npz` 保存实际CPU结果。两组同初始化，各做一次接口检查用的CPU更新，容量仍为64×1536。随机输入与人工张量切片只检查接口，**没有运行视频生成器，不能评价擦除/保护能力或方法优劣**，也不写入正式结果目录。

## 继续研究前的边界

最新两组共享普通门控时序更新器、冻结Wan2.1-VACE-1.3B生成器接入、容量、数据接口和训练设置，区别是配对监督与增加最差读数区域关系项。新项尚可能只是普通损失重加权；TinyHistory尚未接入对照。

真实视频原片、观测标注、实际校准、方法预测视频和方法检查点尚缺。现有原生训练守护器依赖原任务真实预算账本，**不是新机器上的通用训练器**。不要伪造旧账本或ready标记；独立新训练需先建立自己的协议与预算入口。本交接版不启动GPU、不改变原预算。

已测试范围见 [validation/HANDOFF_CHECKS.json](validation/HANDOFF_CHECKS.json)，具体缺口见 [docs/EVIDENCE_STATUS.md](docs/EVIDENCE_STATUS.md)，上游依赖见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
