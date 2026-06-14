import json
import unittest

from utils import process_json_file, clean_screen


class TestUtils(unittest.TestCase):
    def test_process_json_file_minified(self):
        result = process_json_file('data/rico-examples/2.json', minify=True)
        print(result)

    def test_loading_json_binning(self):
        # Testing with different parameters (binning/reduce/minify)
        result = process_json_file('data/rico-examples/2.json',
                                       binning=False,
                                       reduce=True,
                                       minify=False)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result)  # should not throw errors
        print(result)

    def test_process_json_file_pretty(self):
        result = process_json_file('data/rico-examples/2.json', minify=False)
        print(result)

    def test_loading_json_reduce(self):
        # Testing with different parameters (binning/reduce/minify)
        result = process_json_file('data/rico-examples/2.json',
                                       binning=False,
                                       reduce=True,
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
        # Testing with different parameters (binning/reduce/minify)
        result = process_json_file('data/rico-examples/2.json',
                                       binning=False,
                                       reduce=True,
                                       minify=False)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result) # should not throw errors
        print(result)

    def test_json_processing(self):
        # Testing with different parameters (binning/reduce/minify)
        result = process_json_file('data/rico-examples/2.json',
                                       binning=True,
                                       reduce=True,
                                       minify=True)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result) # should not throw errors
        print(result)

if __name__ == '__main__':
    unittest.main()
