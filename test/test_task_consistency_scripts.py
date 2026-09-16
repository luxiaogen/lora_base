import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TaskConsistencyScriptTests(unittest.TestCase):
    def _read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def _assert_qkv_diagnostic_runs(self, text, expected_runs):
        self.assertEqual(text.count("python main.py"), expected_runs)
        self.assertEqual(
            text.count("--set dual_mask_qk_all_tasks=false"),
            expected_runs,
        )
        self.assertEqual(
            text.count("--set dual_mask_qv_all_tasks=false"),
            expected_runs,
        )
        self.assertEqual(
            text.count("--set dual_mask_p_conflict_diagnostics=true"),
            expected_runs,
        )
        self.assertEqual(text.count("--set ca_epochs=5"), expected_runs)
        self.assertNotIn("--set data_path=", text)
        self.assertNotIn('cd "$(dirname "$0")', text)

    def test_5090_script_runs_four_datasets_and_three_seeds(self):
        text = self._read("scripts/9_17_task_consistency_5090_overnight.sh")
        self._assert_qkv_diagnostic_runs(text, 12)
        for config in ("imgr10", "cub10", "imga10", "cifar10"):
            self.assertEqual(
                text.count(f"python main.py --config exps/dlora/{config}.json"),
                3,
            )
        for seed in (1993, 1996, 1997):
            self.assertEqual(text.count(f"--set 'seed=[{seed}]'"), 4)

    def test_3090_children_run_t10_once_and_t20_three_seeds(self):
        t10 = self._read("scripts/9_17_task_consistency_imgr10_3090.sh")
        t20 = self._read("scripts/9_17_task_consistency_imgr20_3090.sh")
        self._assert_qkv_diagnostic_runs(t10, 1)
        self._assert_qkv_diagnostic_runs(t20, 3)
        self.assertEqual(t10.count("--set max_tasks=10"), 1)
        self.assertEqual(t20.count("--set max_tasks=20"), 3)
        for seed in (1993, 1996, 1997):
            self.assertEqual(t20.count(f"--set 'seed=[{seed}]'"), 1)

    def test_3090_wrapper_keeps_data_paths_in_json(self):
        text = self._read("scripts/9_17_task_consistency_3090_overnight.sh")
        self.assertIn(
            "bash scripts/9_17_task_consistency_imgr10_3090.sh",
            text,
        )
        self.assertIn(
            "bash scripts/9_17_task_consistency_imgr20_3090.sh",
            text,
        )
        self.assertNotIn("--set data_path=", text)
        self.assertNotIn('cd "$(dirname "$0")', text)


if __name__ == "__main__":
    unittest.main()
