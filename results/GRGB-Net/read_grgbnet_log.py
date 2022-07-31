import pdb
import json

pdb.set_trace()

dict_images_hard = {}
dict_images_easy = {}

log_file = open("/home/graf-wronski/Projects/dynamic-networks/openmllab/mmclassification/results/GRGB-Net/log2.txt")
lines = log_file.read().split('[')
lines.pop(0) # the first element is empty

for batchblock in lines:
    greyscale_used = batchblock.split(']')[0].replace('\n', '').replace(' ', '').split(',')
    greyscale_used = [x=='True' for x in greyscale_used]

    images = batchblock.split(']')[1].split('\n')
    images.remove('')
    images.remove('')

    for img, b in zip(images, greyscale_used):
        if b == True:
            dict_images_easy[img] = b
        else:
            dict_images_hard[img] = b

with open('results/GRGB-Net/log_grgbnet_easy.json', 'w') as f:
    json.dump(dict_images_easy, f)

with open('results/GRGB-Net/log_grgbnet_hard.json', 'w') as f:
    json.dump(dict_images_hard, f)


log_file.close()