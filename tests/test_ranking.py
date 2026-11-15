from copy import deepcopy
from pathlib import Path
import json
import unittest

from craft_partner_ranking import ranking
from craft_partner_ranking.data import load_bundle, visible_snapshot
from craft_partner_ranking.ranking import candidate_analysis


ROOT = Path(__file__).resolve().parents[1]
FAST = {"draws": 400, "permutations": 200, "validation_folds": 3}


class RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ranking._FIT_CACHE.clear()
        cls.bundle = load_bundle(ROOT / "fixtures/bundle.json")
        cls.snapshot = visible_snapshot(cls.bundle, "2026-01-01")
        cls.analysis = candidate_analysis(cls.snapshot, dict(FAST))

    def test_result_is_finite_and_json_safe(self):
        json.dumps(self.analysis, allow_nan=False, ensure_ascii=False)
        self.assertNotEqual(self.analysis["model"]["method"], "observed-overview-v1")
        self.assertEqual(self.analysis["model"]["method"], "candidate-yield-v1")

    def test_snapshot_is_not_mutated(self):
        original = deepcopy(self.snapshot)
        candidate_analysis(self.snapshot, dict(FAST))
        self.assertEqual(self.snapshot, original)

    def test_every_candidate_has_rank_and_uncertainty(self):
        candidates = self.analysis["result"]["candidates"]
        self.assertEqual(len(candidates), 42)
        for candidate in candidates:
            self.assertIsNotNone(candidate["tier"])
            self.assertGreaterEqual(candidate["rank_interval_80"][1], candidate["rank_interval_80"][0])
            self.assertGreaterEqual(candidate["expected_rate_interval_95"][1], candidate["expected_rate_interval_95"][0])
            self.assertAlmostEqual(
                candidate["expected_achieved"] / candidate["expected_rate"],
                self.analysis["result"]["config"]["slots_per_candidate"],
                places=4,
            )

    def test_small_sample_project_is_shrunk_toward_peers(self):
        candidates = {item["project"]: item for item in self.analysis["result"]["candidates"]}
        small = candidates["project-0"]
        self.assertEqual(small["observed_supported"], 27)
        self.assertGreater(small["expected_rate"], small["observed_rate"])
        shrinkage = {item["project"]: item for item in self.analysis["result"]["borrowing"]["shrinkage"]}
        self.assertGreater(shrinkage["project-0"]["borrowed_fraction"], 0.5)

    def test_organizer_borrowing_dominates_in_this_fixture(self):
        borrowing = self.analysis["result"]["borrowing"]
        self.assertGreater(borrowing["organizer_sd_logit"], borrowing["project_sd_logit"])
        shrinkage = {item["project"]: item for item in borrowing["shrinkage"]}
        self.assertTrue(shrinkage["project-0"]["shares_organizer_with"])

    def test_region_differences_are_retained(self):
        borrowing = self.analysis["result"]["borrowing"]
        self.assertGreaterEqual(len(borrowing["region_effects_logit"]), 3)
        values = list(borrowing["region_effects_logit"].values())
        self.assertGreater(max(values) - min(values), 0.0)

    def test_tiers_are_contiguous_and_cover_candidates(self):
        tiers = self.analysis["result"]["tiers"]
        projects = [project for tier in tiers for project in tier["projects"]]
        self.assertEqual(sorted(projects), sorted(item["project"] for item in self.analysis["result"]["candidates"]))
        self.assertEqual([tier["tier"] for tier in tiers], list(range(1, len(tiers) + 1)))

    def test_portfolio_reports_expected_value_and_risk(self):
        portfolio = self.analysis["result"]["portfolio"]
        self.assertEqual(len(portfolio["selected"]), portfolio["portfolio_size"])
        self.assertGreaterEqual(portfolio["expected_achieved"], portfolio["expected_achieved_interval_80"][0])
        self.assertLessEqual(portfolio["expected_achieved"], portfolio["expected_achieved_interval_80"][1])
        self.assertIn("boundary_tie", portfolio)
        self.assertTrue(all(0 <= value <= 1 for value in portfolio["selection_risk"].values()))
        self.assertGreaterEqual(portfolio["efficiency"], 0.0)
        self.assertLessEqual(portfolio["efficiency"], 1.0 + 1e-9)

    def test_selection_audit_separates_prior_and_post(self):
        audit = self.analysis["result"]["audit"]
        self.assertIn("prior_rate_chosen", audit["retrospective"][0])
        self.assertIn("post_rate_chosen_supported", audit["retrospective"][0])
        # 该资料中入选状态按项目固定，扩容效果不可识别。
        self.assertFalse(audit["selection_identified"])
        conversion = audit["additional_slots_conversion"]
        self.assertEqual(conversion["status"], "not_identified")
        self.assertIsNone(conversion["coefficient"])

    def test_main_exposure_is_supported(self):
        self.assertEqual(self.analysis["model"]["exposure"], "supported")
        self.assertEqual(self.analysis["model"]["exposure_check"]["selected"], "supported")

    def test_variance_components_are_estimated(self):
        priors = self.analysis["model"]["priors"]
        self.assertGreater(priors["organizer_sd_logit"], 0.0)
        self.assertGreater(priors["project_sd_logit"], 0.0)

    def test_config_is_recorded(self):
        config = self.analysis["result"]["config"]
        self.assertIn("seed", config)
        self.assertIn("portfolio_size_source", config)
        self.assertIn("slots_source", config)
        self.assertEqual(
            set(self.analysis["model"]["validation"]["candidates"][0]["configuration"]),
            {"project", "organizer", "region", "category"},
        )


class IdentificationTests(unittest.TestCase):
    """用合成数据验证：能识别的设计才会估计扩容系数。"""

    def build(self, switchers, slot_variation):
        bundle = load_bundle(ROOT / "fixtures/bundle.json")
        nominations = bundle["tables"]["nominations"]
        if switchers:
            # 让一个项目在最后一届改变入选状态，但追加名额仍为 0，避免同时改变名额。
            for row in nominations:
                if row["project"] == "project-2" and row["season"] == 5:
                    row["chosen"] = True
                    row["additional_slots"] = 0
        if slot_variation:
            # 让一个始终入选的项目在最后一届改变追加名额，入选状态不变。
            for row in nominations:
                if row["project"] == "project-0":
                    row["chosen"] = True
                    row["additional_slots"] = 4
                if row["project"] == "project-0" and row["season"] == 5:
                    row["additional_slots"] = 9
        return visible_snapshot(bundle, "2026-01-01")

    def test_fixed_selection_is_not_identified(self):
        snapshot = self.build(False, False)
        analysis = candidate_analysis(snapshot, {"draws": 300, "permutations": 50, "validation_folds": 2})
        self.assertFalse(analysis["result"]["audit"]["selection_identified"])
        self.assertEqual(analysis["result"]["audit"]["additional_slots_conversion"]["status"], "not_identified")

    def test_only_one_source_of_variation_is_not_identified(self):
        switcher_only = candidate_analysis(
            self.build(True, False), {"draws": 300, "permutations": 50, "validation_folds": 2}
        )
        self.assertFalse(switcher_only["result"]["audit"]["selection_identified"])
        slots_only = candidate_analysis(
            self.build(False, True), {"draws": 300, "permutations": 50, "validation_folds": 2}
        )
        self.assertFalse(slots_only["result"]["audit"]["selection_identified"])

    def test_project_switching_allows_estimation(self):
        snapshot = self.build(True, True)
        analysis = candidate_analysis(snapshot, {"draws": 300, "permutations": 50, "validation_folds": 2})
        audit = analysis["result"]["audit"]
        self.assertTrue(audit["selection_identified"])
        self.assertEqual(audit["additional_slots_conversion"]["status"], "estimated")
        self.assertIsNotNone(audit["additional_slots_conversion"]["interval"])


class DataSafetyTests(unittest.TestCase):
    def test_contradictory_rounds_are_rejected(self):
        bundle = load_bundle(ROOT / "fixtures/bundle.json")
        bundle["tables"]["rounds"][0]["achieved"] = 99
        snapshot = visible_snapshot(bundle, "2026-01-01")
        with self.assertRaises(ValueError):
            candidate_analysis(snapshot, {"draws": 300, "permutations": 50, "validation_folds": 2})

    def test_unsupported_correction_field_is_rejected(self):
        bundle = load_bundle(ROOT / "fixtures/bundle.json")
        bundle["tables"]["corrections"].append(
            {
                "id": "correction-x",
                "revision": 1,
                "released_at": "2026-01-01",
                "target": "0-0",
                "field": "project",
                "value": "project-1",
                "reason": "测试",
            }
        )
        snapshot = visible_snapshot(bundle, "2026-01-01")
        with self.assertRaises(ValueError):
            candidate_analysis(snapshot, {"draws": 300, "permutations": 50, "validation_folds": 2})

    def test_insufficient_history_still_returns_candidates(self):
        snapshot = visible_snapshot(load_bundle(ROOT / "fixtures/bundle.json"), "2024-01-11")
        analysis = candidate_analysis(snapshot, {"draws": 300, "permutations": 50, "validation_folds": 2})
        self.assertEqual(analysis["model"]["validation"]["status"], "insufficient_history")
        self.assertTrue(analysis["result"]["candidates"])


if __name__ == "__main__":
    unittest.main()
