import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import main


ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo_cases"


class VariantApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_reports = tempfile.TemporaryDirectory()
        cls._original_reports_dir = main.REPORTS_DIR
        main.REPORTS_DIR = cls._tmp_reports.name
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls) -> None:
        main.REPORTS_DIR = cls._original_reports_dir
        cls._tmp_reports.cleanup()

    def _post_file(self, endpoint: str, path: Path):
        with path.open("rb") as fh:
            return self.client.post(
                endpoint,
                files={"file": (path.name, fh, "text/plain")},
            )

    def test_service_endpoints_are_ready(self) -> None:
        health = self.client.get("/health")
        ready = self.client.get("/ready")
        model_info = self.client.get("/model-info")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")
        self.assertTrue(all(ready.json()["checks"].values()))

        self.assertEqual(model_info.status_code, 200)
        body = model_info.json()
        self.assertEqual(body["model_type"], "XGBClassifier")
        self.assertIn("feature_columns", body)
        self.assertIn("training_summary", body)

    def test_frontend_and_demo_case_routes_work(self) -> None:
        index = self.client.get("/")
        demos = self.client.get("/demo-cases")
        demo_report = self.client.post("/demo-cases/high_signal/report")

        self.assertEqual(index.status_code, 200)
        self.assertIn("Genome Triage Desk", index.text)

        self.assertEqual(demos.status_code, 200)
        demo_items = demos.json()["demo_cases"]
        self.assertTrue(any(item["id"] == "high_signal" for item in demo_items))

        self.assertEqual(demo_report.status_code, 200)
        body = demo_report.json()
        self.assertEqual(body["status"], "completed")
        self.assertIn("clinical_report", body)
        self.assertEqual(
            body["clinical_report"]["overview"]["overall_alert_level"],
            "very_high",
        )

    def test_high_signal_analysis_returns_risks_with_evidence(self) -> None:
        response = self._post_file("/analyze", DEMO_DIR / "high_signal.vcf")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["dangerous_variants_found"], 5)
        self.assertEqual(body["analysis_summary"]["overall_alert_level"], "very_high")
        self.assertEqual(body["follow_up_guidance"]["urgency"], "very_high")
        self.assertGreater(len(body["top_risks"]), 0)

        first = body["top_risks"][0]
        self.assertEqual(first["gene"], "ABCA4")
        self.assertIn("confidence_level", first)
        self.assertIn("confidence_reason", first)
        self.assertIn("reference_model_conflict", first)
        self.assertIn("evidence_profile", first)
        self.assertEqual(first["evidence_profile"]["model_signal"], "very_strong")
        self.assertTrue(first["evidence_profile"]["reference_match"])

    def test_low_signal_analysis_returns_watch_candidates(self) -> None:
        response = self._post_file("/analyze", DEMO_DIR / "low_signal.vcf")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["dangerous_variants_found"], 0)
        self.assertEqual(body["analysis_summary"]["overall_alert_level"], "low")
        self.assertEqual(body["follow_up_guidance"]["urgency"], "low")
        self.assertEqual(body["top_risks"], [])
        self.assertGreater(len(body["top_candidates"]), 0)

        first = body["top_candidates"][0]
        self.assertFalse(first["reference_model_conflict"])
        self.assertEqual(first["evidence_profile"]["population_signal"], "high_frequency")

    def test_report_endpoint_persists_and_can_be_reloaded(self) -> None:
        response = self._post_file("/analyze/report", DEMO_DIR / "mixed_signal.vcf")
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertIn("report_storage", body)
        self.assertIn("clinical_report", body)

        report_id = body["report_storage"]["report_id"]
        self.assertTrue(report_id.startswith("report_"))

        fetch = self.client.get(f"/reports/{report_id}")
        self.assertEqual(fetch.status_code, 200)

        saved = fetch.json()
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["report_id"], report_id)
        self.assertIn("clinical_report", saved)
        self.assertEqual(
            saved["clinical_report"]["overview"]["overall_alert_level"],
            "very_high",
        )

        download = self.client.get(f"/reports/{report_id}/download")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.headers["content-type"], "application/json")


if __name__ == "__main__":
    unittest.main()
