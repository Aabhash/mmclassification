import json
from pathlib import Path


# Opening JSON file

for ending in ["100", "110", "101", "111", "010", "001", "011"]:
    f = open(f'/home/graf-wronski/Projects/dynamic-networks/openmllab/mmclassification/work_dirs/branchynet_resnet50/Experiments/Cifar-10_15-Epochs/validation_branchynet{ending}.json')
    data = json.load(f)
    print(f"Setup: {ending}-BranchyNet")
    for key in data.keys():
        if isinstance(data[key], float):
            print(key)
            print(data[key])
        else:
            print(key)
            # print(data[key][0:2])
    f.close()    
  
# Closing file

