import json
import unittest
import numpy as np

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
        self.test_embedding = np.random.randn(128, 4096).astype(np.float16)
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


if __name__ == '__main__':
    unittest.main()
