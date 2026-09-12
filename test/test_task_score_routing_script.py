import json
import os
import shlex
import subprocess
import unittest


class TaskScoreRoutingScriptTests(unittest.TestCase):
    def test_screen_runs_one_global_model_and_logs_all_modes(self):
        result = subprocess.run(
            ["bash", "scripts/9_12_task_score_routing_screen.sh", "--dry-run"],
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = []
        for line in result.stdout.splitlines():
            parts = shlex.split(line)
            if "main.py" not in parts:
                continue
            args = {}
            for index, part in enumerate(parts):
                if part == "--set":
                    key, value = parts[index + 1].split("=", 1)
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    args[key] = value
            runs.append(args)

        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run["max_tasks"], 3)
        self.assertEqual(run["classification_training_mode"], "task_local")
        self.assertEqual(run["classification_inference_mode"], "global")
        self.assertTrue(run["classification_inference_diagnostics"])
        self.assertFalse(run["dual_mask_task_bias_calibration"])
        self.assertFalse(run["dual_mask_boundary_calibration"])
        self.assertNotIn("data_path", run)


if __name__ == "__main__":
    unittest.main()
