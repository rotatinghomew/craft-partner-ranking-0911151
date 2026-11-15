from __future__ import annotations

from .product import summarize
from .ranking import METHOD as RANKING_METHOD
from .ranking import candidate_analysis


def publish_overview(client, as_of, request_key):
    snapshot = client.snapshot(as_of)
    policy = client.policy()
    artifact = {
        "as_of": as_of,
        "snapshot_id": snapshot["snapshot_id"],
        "method": "observed-overview-v1",
        "result": summarize(snapshot),
    }
    return client.publish(request_key, artifact, policy["revision"])


def publish_candidates(client, as_of, request_key, config=None):
    """计算并发布候选排序，同时保存当时的模型、资料依据与配置。"""
    snapshot = client.snapshot(as_of)
    policy = client.policy()
    analysis = candidate_analysis(snapshot, config)
    artifact = {
        "as_of": as_of,
        "snapshot_id": snapshot["snapshot_id"],
        "method": RANKING_METHOD,
        "result": analysis["result"],
        "model": analysis["model"],
        "data_basis": analysis["data_basis"],
    }
    return client.publish(request_key, artifact, policy["revision"])
