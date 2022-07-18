from os import renames
import torch
from collections import OrderedDict



def main(
    file_1 = "/home/till/mmclassification/pretrained_models/resnet18_b16x8_cifar10_20210528-bd6371c8.pth",
    file_2 = "/home/till/mmclassification/pretrained_models/resnet101_b16x8_cifar10_20210528-2d29e936.pth",
    renames_1 = ["little", "little_head"],
    renames_2 = ["big", "big_head"],
    save_new = "/home/till/mmclassification/pretrained_models/cascading_resnet18_resnet_101.pth"
):
    model_1 = torch.load(file_1)
    save_mode = model_1
    od = model_1['state_dict']
    #new = OrderedDict([(k.replace("backbone", "little"),v) for k,v in od.items()])
    model_1 = OrderedDict([(k.replace("backbone", renames_1[0]).replace("head", renames_1[1]),v) for k,v in od.items()])
    model_2 = torch.load(file_2)
    od = model_2['state_dict']
    #new = OrderedDict([(k.replace("backbone", "little"),v) for k,v in od.items()])
    model_2 = OrderedDict([(k.replace("backbone", renames_2[0]).replace("head", renames_2[1]),v) for k,v in od.items()])
    model_1.update(model_2)
    save_mode['state_dict'] = model_1
    torch.save(save_mode, save_new)


if __name__ == '__main__':
    main()