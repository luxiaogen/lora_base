import json
import os
import shlex
import subprocess
import unittest


class RawTaskScoreScriptTests(unittest.TestCase):
    def test_screen_is_global_read_only_diagnostic(self):
        result = subprocess.run(
            ["bash", "scripts/9_12_raw_task_score_diagnostic.sh", "--dry-run"],
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
        self.assertEqual(run["classification_inference_mode"], "global")
        self.assertFalse(run["classification_inference_diagnostics"])
        self.assertTrue(run["classification_task_score_distribution_diagnostics"])
        self.assertNotIn("data_path", run)


if __name__ == "__main__":
    unittest.main()
