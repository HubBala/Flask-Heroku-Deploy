import json 
with open ("datasetB_sample.json", "r") as f:
    data = json.load(f)

print(data.keys())