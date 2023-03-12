import json

with open("data/vg/region_descriptions.json", "r") as f:
    region_descriptions = json.load(f)

print(region_descriptions[0])
