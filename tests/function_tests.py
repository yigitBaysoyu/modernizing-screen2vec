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


class TestScreenObservability(unittest.TestCase):
    """Whitelist / blacklist scoring in screen_observability.py."""

    @classmethod
    def setUpClass(cls):
        cls.so = importlib.import_module("screen_observability")

    @staticmethod
    def _screen(*components):
        return list(components)

    def test_family_matching(self):
        fam = self.so.family_of
        self.assertEqual(fam("android.webkit.WebView"), "webview")
        self.assertEqual(fam("org.apache.cordova.engine.SystemWebView"), "webview")
        self.assertEqual(fam("org.xwalk.core.internal.XWalkViewBridge"), "webview")
        self.assertEqual(fam("com.google.android.gms.ads.AdView"), "ad")
        self.assertEqual(fam("com.google.android.gms.ads.internal.webview.o"), "ad")
        self.assertEqual(fam("com.mopub.mobileads.MoPubView"), "ad")
        self.assertEqual(fam("com.google.maps.api.android.lib6.impl.am"), "map")
        self.assertEqual(fam("maps.ad.J"), "map")
        self.assertEqual(fam("com.unity3d.player.UnityPlayer"), "engine")
        self.assertEqual(fam("com.adobe.air.AIRWindowSurfaceView"), "engine")
        self.assertEqual(fam("android.view.SurfaceView"), "surface")
        self.assertEqual(fam("android.widget.VideoView"), "surface")
        self.assertEqual(fam("com.google.android.youtube.player.YouTubePlayerView"), "surface")
        # Suffixes match CamelCase tokens, not raw characters.
        self.assertEqual(fam("com.pandora.android.activity.NowPlayingAdView"), "ad")
        self.assertEqual(fam("lam.justbecause.tripcheckerfree.OTCMapView"), "map")
        self.assertEqual(fam("com.mopub.mraid.MraidBridge$MraidWebView"), "ad")
        self.assertEqual(fam("com.carezone.careapp.ui.view.TextureVideoView"), "surface")
        self.assertEqual(self.so.camel_tokens("GLSurfaceView"), ("gl", "surface", "view"))
        # Native views, Chrome's own toolbar and look-alike names must not be opaque.
        for cls in ("android.widget.TextView", "android.widget.Button",
                    "com.facebook.react.views.view.ReactViewGroup",
                    "org.chromium.chrome.browser.omnibox.UrlBar",
                    "ti.modules.titanium.ui.widget.TiUILabel$1",
                    "de.smartchord.droid.tuner.GuitarHeadView",
                    "com.kdanmobile.pdfreader.widget.ReadView.ReadView",
                    "com.modiface.widgets.BitmapView",
                    "com.google.android.apps.zagat.views.LoadView", None, ""):
            self.assertIsNone(fam(cls), cls)

    def test_full_frame_webview_is_fully_covered(self):
        row = self.so.analyse_components(self._screen(
            {"class": "android.webkit.WebView", "bounds": [1, 1, 100, 100]}))
        self.assertAlmostEqual(row["opaque_coverage"], 1.0)
        self.assertAlmostEqual(row["cov_webview"], 1.0)
        self.assertEqual(row["opaque_subtree_count"], 1)
        self.assertEqual(row["largest_opaque_class"], "android.webkit.WebView")

    def test_half_frame_webview(self):
        row = self.so.analyse_components(self._screen(
            {"class": "android.webkit.WebView", "bounds": [1, 1, 100, 50]},
            {"class": "android.widget.TextView", "text": "hello", "bounds": [1, 51, 100, 100]}))
        self.assertAlmostEqual(row["opaque_coverage"], 99 * 49 / (99 * 99))
        self.assertEqual(row["text_component_count"], 1)
        self.assertEqual(row["clean_component_count"], 2)

    def test_overlapping_opaque_boxes_are_a_union_not_a_sum(self):
        row = self.so.analyse_components(self._screen(
            {"class": "android.webkit.WebView", "bounds": [1, 1, 100, 100]},
            {"class": "android.view.SurfaceView", "bounds": [1, 1, 100, 100]}))
        self.assertAlmostEqual(row["opaque_coverage"], 1.0)
        self.assertAlmostEqual(row["cov_webview"], 1.0)
        self.assertAlmostEqual(row["cov_surface"], 1.0)

    def test_webview_with_exposed_text_is_not_opaque(self):
        row = self.so.analyse_components(self._screen(
            {"class": "android.webkit.WebView", "bounds": [1, 1, 100, 100],
             "children": [{"class": "android.view.View", "text": "Sign in", "bounds": [10, 10, 50, 20]}]}))
        self.assertEqual(row["opaque_coverage"], 0.0)
        self.assertEqual(row["opaque_exempt_count"], 1)
        self.assertEqual(row["clean_component_count"], 2)

    def test_ad_with_unlabelled_wrappers_is_opaque(self):
        row = self.so.analyse_components(self._screen(
            {"class": "com.google.android.gms.ads.AdView", "bounds": [1, 87, 100, 93],
             "children": [{"class": "com.google.android.gms.ads.internal.bp", "bounds": [1, 87, 100, 93],
                           "children": [{"class": "in", "bounds": [1, 87, 100, 93]}]}]}))
        self.assertAlmostEqual(row["cov_ad"], 99 * 6 / (99 * 99))
        self.assertEqual(row["opaque_subtree_count"], 1)
        self.assertEqual(row["clean_component_count"], 3)   # wrappers still count as components

    def test_assign_lists_threshold_and_empty(self):
        import pandas as pd
        frame = pd.DataFrame({
            "clean_component_count": [0, 5, 5, 5],
            "opaque_coverage": [0.0, 0.0, 0.49, 0.5],
        })
        out = self.so.assign_lists(frame, threshold=0.5)
        self.assertEqual(out["list"].tolist(), ["blacklist", "whitelist", "whitelist", "blacklist"])
        self.assertEqual(out["reason"].tolist(), ["empty", "", "", "opaque"])

    def test_example_screens(self):
        row0 = self.so.scan_screen("data/rico-examples/0.json")   # Cordova PWA: one WebView
        self.assertEqual(row0["app"], "com.brc.PeriodTrackerDiary")
        self.assertGreater(row0["opaque_coverage"], 0.9)
        self.assertEqual(row0["largest_opaque_class"], "org.apache.cordova.engine.SystemWebView")
        row2 = self.so.scan_screen("data/rico-examples/2.json")   # native settings screen
        self.assertEqual(row2["opaque_coverage"], 0.0)
        self.assertGreater(row2["text_component_count"], 0)

    def test_run_scan_on_examples(self):
        frame, meta = self.so.run_scan(layouts_dir="data/rico-examples", workers=2)
        self.assertEqual(frame.loc[0, "list"], "blacklist")
        self.assertEqual(frame.loc[0, "reason"], "opaque")
        self.assertEqual(frame.loc[2, "list"], "whitelist")
        self.assertNotIn("_class_counter", frame.columns)
        self.assertEqual(meta["screens"], len(frame))
        self.assertEqual(meta["threshold"], 0.5)


class TestDimensionSweep(unittest.TestCase):
    """Naming, log parsing and output writing of 4_sweep_embedding_dimensions.py."""

    @classmethod
    def setUpClass(cls):
        cls.sweep = importlib.import_module("4_sweep_embedding_dimensions")

    def test_recipe_point_keeps_canonical_names(self):
        art = self.sweep.artifact_paths(768, "", self.sweep.artifact_tag(1024, 42))
        self.assertEqual(art["emb_dir"], "embeddings_seq_768")
        self.assertEqual(art["app_dir"], "embeddings_seq_768_with_app")
        self.assertEqual(art["lstm"], "lstm_predictor_768d.pth")
        self.assertEqual(art["projector"], "sequence_projector_768d.pth")

    def test_protocol_hidden_and_seed_suffixes(self):
        proto = self.sweep.protocol_suffix("strict", None)
        art = self.sweep.artifact_paths(128, proto, self.sweep.artifact_tag(4096, 43))
        self.assertEqual(art["emb_dir"], "embeddings_seq_128_strict_h4096_s43")
        self.assertEqual(art["lstm"], "lstm_predictor_128d_strict_h4096_s43.pth")
        self.assertEqual(self.sweep.protocol_suffix("nback", 2), "_nback2")
        self.assertEqual(self.sweep.protocol_flags("nback", 2), ["--nback", "2"])
        self.assertEqual(self.sweep.protocol_flags("permissive", None), [])

    def test_commands_pass_protocol_to_trainer_and_evaluator(self):
        parser = self.sweep.build_parser()
        args = parser.parse_args(["run", "--protocol", "strict", "--dims", "64", "--seeds", "43",
                                  "--hidden_dim", "1024"])
        commands = dict(self.sweep.build_commands(64, args, 43))
        self.assertIn("--strict", commands["train"])
        self.assertIn("--strict", commands["eval"])
        self.assertIn("--artifact_tag", commands["train"])
        self.assertEqual(commands["train"][commands["train"].index("--artifact_tag") + 1], "_s43")
        self.assertIn("lstm_predictor_64d_strict_s43.pth", commands["eval"])
        self.assertNotIn("--strict", commands["app"])
        # Default width is 4096 so that D=2048 fits; it tags every artefact uniformly.
        default_args = parser.parse_args(["run", "--dims", "2048"])
        self.assertEqual(default_args.protocol, "strict")
        default_cmds = dict(self.sweep.build_commands(2048, default_args, 42))
        self.assertEqual(default_cmds["train"][default_cmds["train"].index("--hidden_dim") + 1], "4096")
        self.assertIn("--strict", default_cmds["train"])
        self.assertIn("lstm_predictor_2048d_strict_h4096.pth", default_cmds["eval"])
        permissive = parser.parse_args(["run", "--protocol", "permissive", "--dims", "2048"])
        self.assertIn("lstm_predictor_2048d_h4096.pth", dict(self.sweep.build_commands(2048, permissive, 42))["eval"])
        self.assertEqual(self.sweep.DEFAULT_DIMS[0], 16)
        self.assertEqual(self.sweep.DEFAULT_DIMS[-1], 2048)

    def test_parse_eval_metrics(self):
        text = (
            "Model  Top-1  Top-0.01%  Top-0.1%  Top-1%  Top-5%  Norm.RMSE\n"
            "SWEEP_D64_S42        0.1182     0.5004     0.9917   1.0000   1.0000     0.6028\n"
            "Screen2Vec-EUCLIDEAN-spatial   0.0610 0.2580 0.9690 0.9980 1.0000 0.8530\n"
            "  Supplementary (not in Table 3): MRR 0.2345\n"
            "  Eligible traces evaluated: 5104 (41553 screens, 34140 prediction instances)\n"
        )
        m = self.sweep.parse_eval_metrics(text, "SWEEP_D64_S42")
        self.assertAlmostEqual(m["top1"], 0.1182)
        self.assertAlmostEqual(m["nrmse"], 0.6028)
        self.assertAlmostEqual(m["mrr"], 0.2345)
        self.assertEqual(m["n_traces"], 5104)
        self.assertEqual(m["n_instances"], 34140)
        with self.assertRaises(RuntimeError):
            self.sweep.parse_eval_metrics(text, "SWEEP_D32_S42")

    def test_outputs_from_synthetic_results(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.sweep.SweepPaths("permissive", None, 1024, sweep_dir=Path(directory))
            base = {"top001pct": 0.5, "top01pct": 0.99, "top1pct": 1.0, "top5pct": 1.0,
                    "nrmse": 0.6, "n_traces": 10, "n_screens_in_traces": 50, "n_instances": 30,
                    "completed_at": "now"}
            results = {"meta": {"protocol": "permissive", "hidden_dim": 1024, "seeds": [42, 43]},
                       "runs": {"64": {"42": dict(base, top1=0.10, mrr=0.20, seed=42),
                                       "43": dict(base, top1=0.12, mrr=0.22, seed=43)},
                                "768": {"42": dict(base, top1=0.15, mrr=0.30, seed=42)}}}
            self.sweep.save_results(paths, results)
            self.assertEqual(self.sweep.load_results(paths)["runs"]["768"]["42"]["top1"], 0.15)
            stats = self.sweep.seed_stats(results)
            self.assertAlmostEqual(stats["64"]["top1"]["mean"], 0.11)
            self.assertEqual(stats["768"]["n"], 1)
            table = self.sweep.write_table(paths, results)
            self.assertIn("0.1100+-", table)
            self.sweep.write_csv(paths, results)
            self.sweep.write_figure(paths, results)
            self.assertTrue(paths.figure.exists())
            self.assertEqual(len(paths.csv.read_text(encoding="utf-8").strip().splitlines()), 4)

    def test_noapp_command_uses_unconcatenated_embeddings(self):
        parser = self.sweep.build_parser()
        args = parser.parse_args(["run", "--hidden_dim", "1024", "--dims", "64", "--seeds", "43"])
        cmd = self.sweep.build_noapp_command(64, args, 43)
        self.assertEqual(cmd[cmd.index("--embeddings_dir") + 1], "embeddings_seq_64_strict_s43")
        self.assertNotIn("embeddings_seq_64_strict_s43_with_app", cmd)
        self.assertEqual(cmd[cmd.index("--label") + 1], "SWEEP_D64_S43_NOAPP")
        self.assertIn("--strict", cmd)
        # the with-app label must not match the no-app row and vice versa
        text = "SWEEP_D64_S43_NOAPP   0.0500 0.2000 0.6000 0.9000 0.9700 0.8500\n  MRR 0.1200\n"
        self.assertAlmostEqual(self.sweep.parse_eval_metrics(text, "SWEEP_D64_S43_NOAPP")["top1"], 0.05)
        with self.assertRaises(RuntimeError):
            self.sweep.parse_eval_metrics(text, "SWEEP_D64_S43")

    def test_canonical_point_and_force_guard(self):
        self.assertTrue(self.sweep.is_canonical_point(768, 1024, 42))
        self.assertFalse(self.sweep.is_canonical_point(768, 4096, 42))
        self.assertFalse(self.sweep.is_canonical_point(768, 1024, 43))
        self.assertFalse(self.sweep.is_canonical_point(64, 1024, 42))
        with self.assertRaises(SystemExit) as ctx:
            self.sweep.main(["run", "--hidden_dim", "1024", "--dims", "64,768", "--seeds", "42", "--force", "--dry_run"])
        self.assertIn("lstm_predictor_768d_strict.pth", str(ctx.exception))
        # not canonical -> --force is allowed (dry run prints and returns)
        self.sweep.main(["run", "--hidden_dim", "1024", "--dims", "64", "--seeds", "42", "--force", "--dry_run"])

    def test_reference_row_is_protocol_aware_and_noapp_block(self):
        base = {"top001pct": 0.5, "top01pct": 0.99, "top1pct": 1.0, "top5pct": 1.0, "nrmse": 0.6,
                "n_traces": 10, "n_screens_in_traces": 50, "n_instances": 30, "completed_at": "now"}
        noapp = dict(base, top1=0.04, mrr=0.10, top01pct=0.6)
        runs = {"64": {"42": dict(base, top1=0.10, mrr=0.20, seed=42, no_app=noapp, trained_in_sweep=True),
                       "43": dict(base, top1=0.12, mrr=0.22, seed=43, trained_in_sweep=True)}}
        with tempfile.TemporaryDirectory() as directory:
            for protocol, matched in (("permissive", False), ("strict", True)):
                paths = self.sweep.SweepPaths(protocol, None, 1024, sweep_dir=Path(directory))
                results = {"meta": {"protocol": protocol, "hidden_dim": 1024, "seeds": [42, 43]}, "runs": runs}
                self.assertEqual(self.sweep.reference_is_matched(results["meta"]), matched)
                table = self.sweep.write_table(paths, results)
                self.assertEqual("not protocol-matched" in table, not matched)
                self.assertIn("Without the app-description half", table)
                self.assertIn("no detectable trend", table)
                self.sweep.write_csv(paths, results)
                header, first = paths.csv.read_text(encoding="utf-8").splitlines()[:2]
                self.assertIn("noapp_top1", header)
                self.assertIn("trained_in_sweep", header)
                self.assertIn("0.04", first)
                self.sweep.write_figure(paths, results)
                self.assertTrue(paths.figure.exists())
            stats = self.sweep.seed_stats(results, "no_app")
            self.assertEqual(stats["64"]["n"], 1)
            self.assertAlmostEqual(stats["64"]["top1"]["mean"], 0.04)

