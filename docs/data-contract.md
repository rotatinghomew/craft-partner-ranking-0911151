# 业务资料契约

rounds.supported 是当届获支持机会数，achieved 为达成数，organizer 可同时承办多个项目。nominations 记录事前候选、是否入选以及 additional_slots；known_before 用于区分决策前信息与下届结果。corrections 是字段级复核，在公开后覆盖对应 rounds 的当前有效值，历史发布仍保持原样。projects.region 与 category 是可能的借力层级，并非预先规定的统计模型。

## 资料表

| 表名 | 原始记录数 | 字段 |
| --- | ---: | --- |
| projects | 42 | category, id, organizer, region, released_at, revision |
| rounds | 252 | achieved, budget, id, organizer, project, released_at, revision, season, supported |
| nominations | 252 | additional_slots, candidate, chosen, id, known_before, project, released_at, revision, season |
| corrections | 1 | field, id, reason, released_at, revision, target, value |

记录均为本地合成资料，只表达业务结构，不是公开报道所述人群或机构的实测资料。日期是本地资料日历；所有版本以资料服务的 as_of 为准。字段值和自然缺失保留原义。金额、评分、频率和权重不可跨表不加判断地直接合计。

## 候选排序如何使用这些字段

- **暴露量**：主口径只用 `rounds.supported`。`nominations.additional_slots` 是筛选后才增加的容量，与入选状态同时出现，不能直接并入分母；只有入选状态与追加名额都在项目内变化时，折算系数才可识别，否则标为不可识别。
- **结果**：`achieved` 是 `supported` 个机会中的达成数。资料要求 `0 <= achieved <= supported`；不满足即拒绝，不自行截断。
- **事前与事后**：`known_before` 早于当届 `rounds.released_at`，回看用该日期之前的可见届次比较入选组与对照组，再展示当届结果，避免只用筛选后的表现证明筛选有效。
- **借力层级**：`projects.organizer`、`region`、`category` 是候选的借力结构。地区差异进入固定效应并报告；承办方与项目进入随机效应，收缩强度由资料估计，不按地区一律合并。
- **复核**：`corrections` 在快照内覆盖对应届次的当前有效值，成果的 `data_basis.applied_corrections` 记录实际应用的复核编号。同一 `as_of` 下资料变化会得到新的 `snapshot_id`，不能复用旧成果。
