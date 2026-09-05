import json
import importlib
import tempfile
import unittest
import numpy as np
from pathlib import Path

from utils import process_json_file, clean_screen


class TestUtils(unittest.TestCase):
    def test_process_json_file_minified(self):
        result = process_json_file('data/rico-examples/2.json', minify=True)
        print(result)

    def test_loading_json_binning(self):
        # Testing with minify=False
        result = process_json_file('data/rico-examples/2.json',
                                       minify=False)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result)  # should not throw errors
        print(result)

    def test_process_json_file_pretty(self):
        result = process_json_file('data/rico-examples/2.json', minify=False)
        print(result)

    def test_loading_json_reduce(self):
        # Testing with minify=False
        result = process_json_file('data/rico-examples/2.json',
                                       minify=False)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result)  # should not throw errors
        print(result)

    def test_clean_screen_shape(self):
        screen = clean_screen('data/rico-examples/2.json')
        # cleaned screen is {app, activity, components}
        assert set(screen.keys()) == {"app", "activity", "components"}, screen.keys()
        assert isinstance(screen["components"], list)

    def test_loading_json_minify(self):
        # Testing with minify=False
        result = process_json_file('data/rico-examples/2.json',
                                       minify=False)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result) # should not throw errors
        print(result)

    def test_json_processing(self):
        # Testing with different parameters (binning/reduce/minify)
        # Note: legacy tests parameters. Since utils.py signature changed, we just test basic.
        result = process_json_file('data/rico-examples/2.json', minify=True)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result) # should not throw errors
        print(result)


class TestLoadEmbeddings(unittest.TestCase):
    def setUp(self):
        import os
        import numpy as np
        
        self.test_embeddings_dir = "test_embeddings_dir"
        os.makedirs(self.test_embeddings_dir, exist_ok=True)
        
        # We will create a test embedding for Rico example '0.json'
        self.test_embedding = np.random.randn(4096).astype(np.float16)
        np.savez_compressed(
            os.path.join(self.test_embeddings_dir, "0.npz"),
            embedding=self.test_embedding
        )
        
    def tearDown(self):
        import shutil
        import os
        if os.path.exists(self.test_embeddings_dir):
            shutil.rmtree(self.test_embeddings_dir)
            
    def test_load_embeddings(self):
        from utils import load_embeddings
        df = load_embeddings(
            embeddings_dir=self.test_embeddings_dir,
            data_dir="data/rico-examples"
        )
        self.assertEqual(len(df), 1)
        self.assertIn("json-path", df.columns)
        self.assertIn("jpg-path", df.columns)
        self.assertIn("embedding", df.columns)
        self.assertEqual(df.iloc[0]["json-path"].replace('\\', '/'), "data/rico-examples/0.json")
        # Check that the embedding was loaded correctly as float32
        loaded_emb = df.iloc[0]["embedding"]
        np.testing.assert_array_almost_equal(
            loaded_emb,
            self.test_embedding.astype(np.float32),
            decimal=3
        )


class TestReconstructionRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reconstruction = importlib.import_module("3_json_reconstruction_par")

    def test_oom_batch_is_recursively_split(self):
        items = list(range(8))
        arrays = list(range(8))
        cleanups = []

        def generate(batch):
            if len(batch) > 2:
                raise RuntimeError("CUDA out of memory")
            return [f"result-{item}" for item in batch]

        groups = self.reconstruction.generate_with_oom_splitting(
            items, arrays, generate, cleanup=lambda: cleanups.append(True)
        )

        flattened_items = [item for group, _ in groups for item in group]
        flattened_texts = [text for _, texts in groups for text in texts]
        self.assertEqual(flattened_items, items)
        self.assertEqual(flattened_texts, [f"result-{item}" for item in items])
        self.assertEqual(len(cleanups), 3)

    def test_batch_size_one_oom_is_fatal(self):
        def generate(_):
            raise RuntimeError("CUDA out of memory")

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.reconstruction.generate_with_oom_splitting(
                [1], [1], generate
            )

    def test_error_report_is_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "7_reconstructed.json").write_text("{}", encoding="utf-8")
            (output / "7_report.txt").write_text(
                "ERROR processing RICO element 7\n", encoding="utf-8"
            )
            items = [{"rico_id": 7}]
            self.assertEqual(
                self.reconstruction.pending_items(items, output), items
            )

    def test_audit_requires_complete_non_error_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings"
            output = root / "output"
            embeddings.mkdir()
            output.mkdir()

            for rico_id in range(2):
                (embeddings / f"{rico_id}.npz").write_bytes(b"placeholder")
                (output / f"{rico_id}_reconstructed.json").write_text(
                    "{}", encoding="utf-8"
                )
                (output / f"{rico_id}_report.txt").write_text(
                    "METRICS:\nJSON Valid: Yes\n", encoding="utf-8"
                )

            complete = self.reconstruction.audit_output_dir(output, embeddings, 2)
            self.assertTrue(complete["ok"])

            (output / "1_report.txt").write_text(
                "ERROR processing RICO element 1\n", encoding="utf-8"
            )
            incomplete = self.reconstruction.audit_output_dir(output, embeddings, 2)
            self.assertFalse(incomplete["ok"])
            self.assertEqual(incomplete["error_reports"], 1)


class TestHpc6RecoveryLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = importlib.import_module(
            "hpc._hpc6_resume_qwen3_4b_reconstructions"
        )

    def test_array_uses_forty_tasks_capped_at_eight_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "recover.sh"
            jobs = [root / f"job_{index}.json" for index in range(5)]
            self.launcher.write_array_script(script, root, jobs)
            text = script.read_text(encoding="utf-8")

            self.assertIn("#SBATCH --array=0-39%8", text)
            self.assertIn("#SBATCH --gres=gpu:1", text)
            self.assertIn("run_index=$((task_id % 5))", text)
            self.assertIn("shard_index=$((task_id / 5))", text)
            self.assertIn("--oom_split", text)
            self.assertIn("--require_complete_shard", text)

    def test_17b_last_run_uses_matching_adapter(self):
        run = next(
            run for run in self.launcher.RUNS
            if run["model_slug"] == "qwen3-1.7b-base"
        )
        self.assertEqual(run["variant"], "4096_last")
        self.assertEqual(
            run["adapter_dir"].name,
            "adapter_model_out_embeddings_4096_last_full_qwen3-1.7b-base",
        )
        self.assertEqual(
            run["output_dir"].name,
            "reconstructions_4096_last__via__"
            "embeddings_4096_last_full_qwen3-1.7b-base",
        )


if __name__ == '__main__':
    unittest.main()
