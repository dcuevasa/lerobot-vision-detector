import json
from pathlib import Path
import unittest

import numpy as np

from lerobot_vision_detector.cli.rollout_logging import (
    analyze_and_log_rollout,
    get_synced_policy_cli_args,
    update_run_outcome,
)


class RolloutLoggingTests(unittest.TestCase):
    def test_get_synced_policy_cli_args_respects_overrides(self):
        with self._tempdir() as tmp:
            model_dir = tmp / "xvla_ckpt"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "type": "xvla",
                        "chunk_size": 30,
                        "n_action_steps": 30,
                        "dtype": "bfloat16",
                        "action_mode": "auto",
                        "max_state_dim": 20,
                        "max_action_dim": 20,
                        "resize_imgs_with_padding": [224, 224],
                        "num_image_views": 3,
                        "tokenizer_max_length": 1024,
                        "pretrained_revision": None,
                        "compile_model": False,
                    }
                ),
                encoding="utf-8",
            )

            flags = get_synced_policy_cli_args(
                str(model_dir),
                extra_cli_args=["--policy.n_action_steps=15"],
            )
            joined = "\n".join(flags)
            self.assertIn("--policy.chunk_size=30", joined)
            self.assertNotIn("--policy.n_action_steps=30", joined)
            self.assertIn("--policy.action_mode=auto", joined)
            self.assertIn("--policy.max_state_dim=20", joined)
            self.assertIn("--policy.resize_imgs_with_padding=[224,224]", joined)
            self.assertNotIn("pretrained_revision", joined)
            self.assertNotIn("compile_model", joined)

    def test_analyze_and_log_rollout_generalist_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with self._tempdir() as tmp:
            ds_root = tmp / "local" / "eval_test_run"
            (ds_root / "meta").mkdir(parents=True)
            (ds_root / "data" / "chunk-000").mkdir(parents=True)

            (ds_root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "fps": 30,
                        "features": {
                            "action": {"dtype": "float32", "shape": [4], "names": ["j0", "j1", "j2", "ee"]},
                            "observation.state": {"dtype": "float32", "shape": [4], "names": ["j0", "j1", "j2", "ee"]},
                        },
                    }
                ),
                encoding="utf-8",
            )

            n = 60
            t = np.arange(n, dtype=np.float32) / 30.0
            actions = np.stack(
                [
                    np.linspace(0.0, 40.0, n),
                    np.linspace(-10.0, 20.0, n),
                    np.sin(np.linspace(0.0, 3.0, n)) * 10.0,
                    np.where((np.arange(n) >= 20) & (np.arange(n) < 40), 5.0, 35.0),
                ],
                axis=1,
            ).astype(np.float32)
            # Inject boundary jumps at chunk boundaries (steps 15, 30, 45)
            actions[15:, 0] += 12.0
            actions[30:, 0] += 12.0
            actions[45:, 0] += 12.0
            states = actions - 1.5

            table = pa.table(
                {
                    "observation.state": [row.tolist() for row in states],
                    "action": [row.tolist() for row in actions],
                    "timestamp": t.tolist(),
                    "episode_index": [0] * n,
                    "frame_index": list(range(n)),
                }
            )
            pq.write_table(table, ds_root / "data" / "chunk-000" / "file-000.parquet")

            log_dir = tmp / "eval_logs"
            record = analyze_and_log_rollout(
                dataset_root=ds_root,
                dataset_repo_id="local/eval_test_run",
                policy_path="dummy/model",
                policy_type="xvla",
                n_action_steps=15,
                chunk_size=30,
                task="Move object",
                log_dir=log_dir,
                outcome="success",
                notes="clean run",
                run_id="test_run_001",
            )

            self.assertEqual(record["timing"]["num_frames"], 60)
            self.assertAlmostEqual(record["timing"]["fps_median"], 30.0, places=1)
            self.assertAlmostEqual(record["tracking_lag"]["overall_mean"], 1.5, places=2)
            self.assertEqual(record["kinematics"]["commanded_ee_cycles"], 1)
            self.assertEqual(record["kinematics"]["measured_ee_cycles"], 1)
            self.assertFalse(record["kinematics"]["is_frozen_arm"])
            self.assertGreater(record["kinematics"]["boundary_to_interior_ratio"], 1.5)

            self.assertTrue((ds_root / "rollout_analysis.json").is_file())
            self.assertTrue((ds_root / "rollout_analysis.md").is_file())
            self.assertTrue((log_dir / "runs" / "test_run_001" / "rollout_analysis.json").is_file())
            self.assertTrue((log_dir / "eval_history.jsonl").is_file())
            log_md = (log_dir / "EVAL_RESULTS_LOG.md").read_text(encoding="utf-8")
            self.assertIn("test_run_001", log_md)
            self.assertIn("clean run", log_md)

            updated = update_run_outcome(
                log_dir=log_dir,
                outcome="partial",
                notes="grasped block, dropped 2cm short of pink pad",
            )
            self.assertEqual(updated["outcome"], "partial")
            self.assertIn("dropped 2cm short", (log_dir / "EVAL_RESULTS_LOG.md").read_text(encoding="utf-8"))

    @staticmethod
    def _tempdir():
        import tempfile

        class _Ctx:
            def __enter__(self):
                self._td = tempfile.TemporaryDirectory()
                return Path(self._td.name)

            def __exit__(self, exc_type, exc, tb):
                self._td.cleanup()

        return _Ctx()


if __name__ == "__main__":
    unittest.main()
