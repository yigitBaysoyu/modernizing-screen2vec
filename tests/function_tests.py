import json
import unittest

from utils import process_json_file


class TestUtils(unittest.TestCase):
    def test_loading_json_binning(self):
        # Testing with different parameters (binning/reduce/minify)
        result = process_json_file('data/rico-examples/2.json',
                                       binning=False,
                                       reduce=True,
                                       minify=False)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result) # should not throw errors
        print(result)

    def test_loading_json_reduce(self):
        # Testing with different parameters (binning/reduce/minify)
        result = process_json_file('data/rico-examples/2.json',
                                       binning=False,
                                       reduce=True,
                                       minify=False)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result) # should not throw errors
        print(result)

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