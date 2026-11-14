from collections import defaultdict


def round_ledger(snapshot):
    rounds = {row["id"]: dict(row) for row in snapshot["tables"]["rounds"]}
    for correction in snapshot["tables"]["corrections"]:
        if correction["target"] in rounds:
            if correction["field"] not in {"supported", "achieved", "budget"}:
                raise ValueError("不支持的复核字段")
            rounds[correction["target"]][correction["field"]] = correction["value"]
    for row in rounds.values():
        if row["supported"] <= 0 or not 0 <= row["achieved"] <= row["supported"]:
            raise ValueError("项目达成数与支持数矛盾")
    return list(rounds.values())


def raw_rates(snapshot):
    totals = defaultdict(lambda: [0,0])
    for row in round_ledger(snapshot):
        totals[row["project"]][0] += row["achieved"]
        totals[row["project"]][1] += row["supported"]
    return [{"project": name, "achieved": values[0], "supported": values[1], "observed_rate": values[0]/values[1]} for name, values in sorted(totals.items())]


def summarize(snapshot):
    rows = raw_rates(snapshot)
    return {"raw_order": sorted(rows, key=lambda row: (-row["observed_rate"], row["project"])), "interpretation": "历史比例排序，不表达收缩估计、未来支持收益或选后可信度"}
