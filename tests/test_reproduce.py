"""Compare fresh computations against the frozen study results."""
import unittest
import reproduce


class ReproductionTests(unittest.TestCase):
    def test_interpreted_flood_reference_scores(self):
        result = reproduce.detection()
        self.assertEqual(result["locations"], 52)
        self.assertEqual(result["counts"], {"TP": 8, "TN": 37, "FN": 2, "FP": 2, "uncertain": 3})
        self.assertAlmostEqual(result["raw"]["precision"], .8)
        self.assertAlmostEqual(result["raw"]["recall"], .8)
        self.assertAlmostEqual(result["weighted"]["precision"], 76/81)
        self.assertAlmostEqual(result["weighted"]["recall"], 76/115)

    def test_refitted_reservoir_forecasts(self):
        result = reproduce.reservoir()
        self.assertEqual(result["accepted_records"], {"Bhakra": 1064, "Pong": 830})
        self.assertEqual(result["candidate_fits_checked"], 40)
        self.assertEqual(len(result["metrics"]), 15)
        primary = {(row["dam"], row["horizon_days"]): row for row in result["metrics"]
                   if row["period"] == "primary_2021_2024"}
        self.assertEqual(primary["Bhakra", 1]["n"], 194)
        self.assertAlmostEqual(primary["Bhakra", 1]["mae_m"], .24715554682227403)
        self.assertEqual(primary["Pong", 1]["n"], 12)
        self.assertAlmostEqual(primary["Pong", 1]["mae_m"], .9157080047205616)
        self.assertLess(primary["Pong", 2]["undamped_mae_m"], primary["Pong", 2]["mae_m"])

    def test_refitted_integration_models(self):
        result = reproduce.integration()
        self.assertEqual(result["reconstructed_scores"], 180)
        self.assertLessEqual(result["maximum_score_difference"], 1e-12)
        self.assertEqual(len(result["results"]), 9)
        for row in result["results"]:
            self.assertEqual(row["test_cases"], 20)
            self.assertEqual(row["counts"], {"FN": 10, "TN": 10})


if __name__ == "__main__":
    unittest.main()
