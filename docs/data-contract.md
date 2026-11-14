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
