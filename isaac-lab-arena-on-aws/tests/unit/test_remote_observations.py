"""Prevent completed host checklists disappearing on their way to the caller."""
import unittest

from vla_pipeline.remote_execution import apply_host_observations


class RemoteObservationTests(unittest.TestCase):
    def test_completed_checklist_reaches_caller_without_replacing_its_context(self):
        checks = {"status": "passed", "checks": [{
            "name": "check_training_job_success_log", "status": "passed",
            "description": "The trainer completed the requested step count."}]}
        request = {"transport": "ssm"}
        caller = {"profile": "caller-profile", "request": request, "state_dir": "/caller/state"}
        host = {"status": "Succeeded", "independently_verified": True, "contract_checks": checks,
                "profile": None, "request": {"transport": "local"}, "state_dir": "/host/state"}
        apply_host_observations(caller, host)
        self.assertEqual(caller["contract_checks"], checks)
        self.assertTrue(caller["independently_verified"])
        self.assertEqual(caller["status"], "Succeeded")
        self.assertEqual((caller["profile"], caller["request"], caller["state_dir"]),
                         ("caller-profile", request, "/caller/state"))

    def test_failed_checklist_and_reason_reach_caller_unchanged(self):
        checks = {"status": "failed", "checks": [{
            "name": "check_training_step_count", "status": "failed",
            "reason": "Training stopped at step 199; requested 200."}]}
        caller = {"status": "Running"}
        host = {"status": "VerificationFailed", "independently_verified": False,
                "failure_reason": "Named pipeline checks failed", "contract_checks": checks}
        apply_host_observations(caller, host)
        self.assertEqual(caller["contract_checks"], checks)
        self.assertFalse(caller["independently_verified"])
        self.assertEqual(caller["status"], "VerificationFailed")
        self.assertEqual(caller["failure_reason"], host["failure_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
