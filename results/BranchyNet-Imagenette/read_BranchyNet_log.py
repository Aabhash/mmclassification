import pdb
import json

from numpy import logical_or

''' This file was used for a one-time reading of log data. It was used in combination with some 
    manual adjustments to the log file and can not be simply used on the logging output.
    
    It sorts the logged images in 'easy' (only first exit), 'medium' (first and second exit) 
    and hard (all three exits).'''

dict_images_hard = {}
dict_images_medium = {}
dict_images_easy = {}

log_file = open("/home/graf-wronski/Projects/dynamic-networks/openmllab/mmclassification/results/BranchyNet-Imagenette/log1.txt")
pdb.set_trace()

lines = log_file.read().split(')')
lines.remove('') # the first element is empty

i = 0
while i < len(lines) - 1:
    exits_used = lines[i]
    images = lines[i+1]

    arr = exits_used.replace('True,\n', 'True,').replace('[ True]', 'True').replace('  ', '').replace(',', ', ').replace('\nT', 'T').replace('[False]', 'False').replace('\n[False]', 'False').replace('\n','').replace('[', '').split(']')
    exit1 = arr[0].split(',')
    exit2 = arr[1][3:].split(',')
    exit3 = arr[2][3:].split(',')

    images = images.split('\n')
    images.remove('') # last element is empty

    for img,ex1,ex2 in zip(images, exit1, exit2):
        if ex1 == 'False':
            if ex2 == 'False':
                dict_images_hard[img] = 'hard'
            else:
                dict_images_medium[img] = 'medium'
        dict_images_easy[img] = 'easy'
        
    i += 2

with open('results/BranchyNet-Imagenette/log_BranchyNet_easy.json', 'w') as f:
    json.dump(dict_images_easy, f)

with open('results/BranchyNet-Imagenette/log_BranchyNet_medium.json', 'w') as f:
    json.dump(dict_images_medium, f)

with open('results/BranchyNet-Imagenette/log_BranchyNet_hard.json', 'w') as f:
    json.dump(dict_images_hard, f)


log_file.close()