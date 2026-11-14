from copy import deepcopy
from pathlib import Path
import json
import unittest

from craft_partner_ranking.data import load_bundle, visible_snapshot
from craft_partner_ranking.product import *


class ProductTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.snapshot = visible_snapshot(load_bundle(root / "fixtures/bundle.json"), "2026-01-01")

    def test_overview_is_finite_and_preserves_sources(self):
        original = deepcopy(self.snapshot)
        result = summarize(self.snapshot)
        self.assertTrue(result)
        self.assertIn("interpretation", result)
        json.dumps(result, allow_nan=False)
        self.assertEqual(self.snapshot, original)

    def test_correction_updates_effective_round(self):
        row = next(row for row in round_ledger(self.snapshot) if row["id"] == "4-3")
        self.assertEqual(row["achieved"],7)

    def test_aggregate_rates_are_exposure_weighted(self):
        rows = raw_rates(self.snapshot)
        project = rows[0]
        source = [row for row in round_ledger(self.snapshot) if row["project"] == project["project"]]
        self.assertAlmostEqual(project["observed_rate"],sum(row["achieved"] for row in source)/sum(row["supported"] for row in source))
