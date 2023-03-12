import json
from itertools import product

with open("data/vg/region_descriptions.json", "r") as f:
    region_descriptions = json.load(f)

print(list(product([1, 2, 3], [4, 5, 6])))
