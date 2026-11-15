from pathlib import Path
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from craft_partner_ranking.client import ProductClient
from craft_partner_ranking.data import load_bundle
from craft_partner_ranking.service import running_service
from craft_partner_ranking.store import ArtifactStore
from craft_partner_ranking.workflow import publish_candidates, publish_overview


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = load_bundle(root / "fixtures/bundle.json")
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(str(Path(directory) / "state.sqlite3"), bundle.get("policy"))
        try:
            with running_service(bundle, store) as base:
                client = ProductClient(base)
                overview = publish_overview(client, "2026-01-01", "overview-demo")
                candidates = publish_candidates(client, "2026-01-01", "candidates-demo")
                restored = client.artifact(candidates["artifact_id"])
                assert restored == candidates["artifact"]
                result = restored["result"]
                print(
                    json.dumps(
                        {
                            "产品": bundle["product"],
                            "原始资料概览编号": overview["artifact_id"],
                            "候选排序编号": candidates["artifact_id"],
                            "方法": restored["method"],
                            "资料依据": restored["data_basis"],
                            "组合": result["portfolio"],
                            "并列层级": result["tiers"][:3],
                            "借力结构": {
                                "项目方差": result["borrowing"]["project_sd_logit"],
                                "承办方方差": result["borrowing"]["organizer_sd_logit"],
                                "地区效应": result["borrowing"]["region_effects_logit"],
                            },
                            "筛选回看": result["audit"]["additional_slots_conversion"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        finally:
            store.close()


if __name__ == "__main__":
    main()
