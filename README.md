# 合作项目候选排序与不可改写的支持名单

craft-partner-ranking 是 Python 业务分析应用，提供两种成果：`observed-overview-v1` 只陈述观测记录本身；`candidate-yield-v1` 在同一未来支持名额下比较候选的预期达成与不确定性，并回看过去的筛选与扩容。两者方法名不同、结果结构不同，不能把推断当作观测。

## 目录和边界

`craft_partner_ranking/data.py` 读取资料并冻结公开日期可见的版本。`product.py` 实现观测摘要；`ranking.py` 实现候选排序、借力结构、筛选回看与组合收益；`workflow.py` 串联资料读取与成果发布；`service.py` 提供本地 HTTP 资料和消费接口；`store.py` 保存不可变成果和当前消费约束；`client.py` 是调用适配层。服务接口见 `docs/protocol.md`，具体资料含义见 `docs/data-contract.md`。

`tests/test_product.py`、`tests/test_data.py` 覆盖原有观测口径；`tests/test_ranking.py` 覆盖候选排序的收缩、借力、并列、组合边界、识别条件与资料校验；`tests/test_http.py` 覆盖发布、重放、不可改写和复核后的新分析。

## 候选排序口径

`candidate-yield-v1` 用项目-届次二项模型估计达成概率。候选的排序依据是**同一未来支持名额下的后验预期达成**，不是历史比例。要点：

- **小样本收缩**：项目效应向同承办方与同地区借力，借力强度由资料估计的方差分量决定，不预先规定。每个候选同时给出观测比例、后验均值、后验区间、名次区间和借到的信息比例。
- **借力层级要验证**：在既有届次上做向前验证，只用该届决策前可见的届次训练，比较地区、项目、承办方及其组合结构的样本外对数预测分。分数接近时优先保留地区层级；历史不足以验证时不启用未经检验的收缩。
- **不抹平地区差异**：地区进入固定效应并随成果报告；只有地区样本过少时才被先验收缩。
- **共同承办方**：承办方作为随机效应，报告其方差占比、各承办方效应和每个候选实际共承办的项目，说明信息从哪里借来。
- **组合收益与不稳范围**：在给定入选名额与每个候选名额下，给出组合预期达成、80%/95% 区间、预测区间、事后最优差距、入选集合正确概率、每个名额的选择风险和组合边界是否并列。
- **证据不足就并列**：相邻候选的后验优势概率低于阈值时归入同一层级；候选不再被强行排出确定名次。

### 筛选与扩容回看

`nominations` 的入选状态与 `additional_slots` 是筛选后的过程。回看分三段呈现：

1. 决策前可见届次的入选组/对照组差异，并在承办方层内做置换检验；
2. 当届在原始 `supported` 与扩容后暴露量下的结果，以及两组从决策前到当届的变化（含双重差分）；
3. 扩容折算系数：只有入选状态与追加名额都在项目内变化时才可识别；否则明确标记 `not_identified` 并说明原因，主分析不把追加名额计入暴露量。

主分析固定用 `rounds.supported` 作暴露量；`supported+additional_slots` 只作样本外敏感性对照，只有明确更好时才可能被采用，并写入成果。

## 成果保存

发布成果含 `as_of`、`snapshot_id`、`method`、`result`，以及 `model` 和 `data_basis`。`model` 记录似然、结构、暴露量、方差分量、先验、验证折与分数、暴露量对照、已应用复核、局限；`data_basis` 记录资料日期、指纹、各表记录数和应用到的复核编号。首次成功返回不可变 `artifact_id`；同键异内容为 409，已有成果不会被后续修订覆盖。

后来的项目更正会形成新的 `snapshot_id` 与新分析，**不会改写已经发出的名单**。复核过的资料仍保留旧成果供回看。

## 环境和依赖

兼容 Python 3.11.x，具体范围见 `pyproject.toml`；`.python-version` 固定受支持的次版本，不限定补丁版本。独立 Docker 镜像使用固定摘要的 Python 3.11.14，生产验证容器使用 Python 3.11.2，两者采用相同的完整锁定依赖。`requirements.lock` 与 `requirements.txt` 固定数值计算和模型分析的完整依赖版本。应用运行、测试及资料均在本地，不请求外部 API；构建时预装依赖，正式运行不安装或下载资料。

```bash
docker build -t craft-partner-ranking:local .
docker run --rm --network none craft-partner-ranking:local python tools/demo.py
docker run --rm --network none craft-partner-ranking:local python -m unittest discover -s tests -v
```

已有 Python 环境可以先执行 `python -m pip install -r requirements.lock`，然后执行以下检查与产品演示：

```bash
python -m compileall -q craft_partner_ranking tests tools
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
python tools/demo.py
```

演示会启动真正的本地 HTTP 服务，读取业务资料，发布概览和候选排序，读回成果并打印组合、并列层级、借力结构与筛选回看，结束时自动退出。每次使用临时数据库，不覆盖工作目录。

## 交互服务

```bash
python -m craft_partner_ranking.service --port 8765
```

默认服务地址为 `127.0.0.1:8765`，本地成果保存在 `local-state.sqlite3`。启动后资料只读；本地消费约束可通过管理接口变更，不会修改已发布成果。中断运行可正常退出。额外分析使用自己的方法名和结果结构，不能复用旧方法名掩盖语义变化。
