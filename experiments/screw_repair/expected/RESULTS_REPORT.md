# 规范状态＋受保护历史证据：直接对照实际结果

冻结判据输出：**STOP_NO_EXTRA_BENEFIT_IN_THIS_PILOT**。完整性和主张审阅另行记录。

每格为两个新测试draw、三个模型种子的描述均值；相同家族/载荷/种子相关，不能当独立事故。

|方法|测试操作|MAE mm|危险误放行|维修执行超阈值|真实维修regret|等价决策分歧|原任务过关单元|
|---|---|---:|---:|---:|---:|---:|---:|
|tf_plain|seen_repair|0.03702|12.08%|5.26%|0.44057|0.74%|0/6|
|tf_plain|partial_patch|0.04355|4.97%|1.35%|0.15818|0.76%|1/6|
|tf_plain|brace_install|0.02379|36.80%|1.52%|0.15974|1.01%|0/6|
|tf_pair|seen_repair|0.03731|6.63%|4.67%|0.39161|0.81%|0/6|
|tf_pair|partial_patch|0.04660|2.36%|1.74%|0.20096|0.75%|1/6|
|tf_pair|brace_install|0.02137|15.44%|0.63%|0.07277|0.67%|1/6|
|tf_scope|seen_repair|0.04359|10.85%|5.00%|0.41671|1.12%|0/6|
|tf_scope|partial_patch|0.05260|13.39%|1.81%|0.20804|1.20%|0/6|
|tf_scope|brace_install|0.01899|32.56%|1.30%|0.13663|0.80%|1/6|
|canonical_scope|seen_repair|0.04551|9.15%|5.31%|0.44955|1.25%|0/6|
|canonical_scope|partial_patch|0.04912|3.61%|1.37%|0.16281|1.46%|2/6|
|canonical_scope|brace_install|0.04033|2.14%|1.27%|0.15044|1.94%|3/6|
|decoded_pair|seen_repair|0.01561|3.19%|2.49%|0.16629|1.76%|0/6|
|decoded_pair|partial_patch|0.01834|0.10%|0.66%|0.07731|0.86%|4/6|
|decoded_pair|brace_install|0.01756|8.71%|0.35%|0.04282|0.47%|4/6|

|对照|操作|维修regret相对下降|差值bootstrap上界|非劣regret单元|额外收益筛查|
|---|---|---:|---:|---:|---|
|tf_pair|partial_patch|18.99%|-0.00135|5/6|False|
|tf_pair|brace_install|-106.73%|0.12416|1/6|False|
|tf_scope|partial_patch|21.74%|0.00152|4/6|False|
|tf_scope|brace_install|-10.11%|0.06371|3/6|False|
|tf_plain|partial_patch|-2.92%|0.03812|4/6|False|
|tf_plain|brace_install|5.82%|0.03855|3/6|False|
|decoded_pair|partial_patch|-110.60%|0.12689|2/6|False|
|decoded_pair|brace_install|-251.35%|0.15144|1/6|False|

本轮是论文canonical+witness结构的连续响应适配，未增加私有状态或pre-action标签，省略所有方法的pre-action辅助损失。普通scope对照使用同一erase/protect损失以区分结构和目标重加权。

视觉方法共用同一冻结SigLIP2特征。原plain/paired以及公开RGB读数GRU复用已审计冻结训练，新增24条完整轨迹；全部同数据、更新、2LR搜索与验证选择。不同结构容量/FLOPs、数字读数与视觉表征不声称相同。

使用1024个全新基础家族，排除前轮所有基础元组；操作面板共享家族且等价终态可重复。旧测试属于开发证据，没有用旧模型测试预测改本轮协议。

实际选择在私有真实状态副本上重新积分执行。0.35mm阈值、固定成本和无拒绝执行选项属于仿真utility。Bootstrap先平均技术种子再按家族分draw重采样；主要4个regret对比使用98.75%单侧分位上界，只是有限仿真筛查。

没有真实螺丝、硬件标定、自然视频、主动选测或基础世界模型结果。本轮负面结果仅约束这一固定实现/任务/预算。
