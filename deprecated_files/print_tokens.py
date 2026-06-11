# This file prints some tokens from the jsons

import json
from utils import Config, parse_rico_json


config = Config()


if __name__ == '__main__':
    
    # Load JSONS
    dataset = [f"data/rico-examples/{i}.json" for i in range(1)]
    
    for i, data in enumerate(dataset):
        layout_tokens, json_tokens = parse_rico_json(data, config)
        print(f"Layout tokens: {layout_tokens}")
        print(f"Json tokens: {json_tokens}")
        