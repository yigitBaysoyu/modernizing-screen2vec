import json
import unittest

from utils import process_json_file, clean_screen


class TestUtils(unittest.TestCase):
    def test_process_json_file_minified(self):
        result = process_json_file('data/rico-examples/2.json', minify=True)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result)  # should not throw errors
        print(result)

    def test_process_json_file_pretty(self):
        result = process_json_file('data/rico-examples/2.json', minify=False)
        assert isinstance(result, str), f"Result is not a string: {result}"
        _ = json.loads(result)  # should not throw errors
        print(result)

    def test_clean_screen_shape(self):
        screen = clean_screen('data/rico-examples/2.json')
        # cleaned screen is {app, activity, components}
        assert set(screen.keys()) == {"app", "activity", "components"}, screen.keys()
        assert isinstance(screen["components"], list)


if __name__ == '__main__':
    unittest.main()
