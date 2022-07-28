import argparse
from ast import arg
from tkinter import Y
from unittest import result
import matplotlib.pyplot as plt

from mmcv import Config
from collections import defaultdict
import json

from torchinfo import summary
from mmcls.models import build_classifier
from statistics import mean

def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('results_configs', help='config file path')
    parser.add_argument('all_configs', help='config file path')
    args = parser.parse_args()
    return args
def main():
    #args = parse_args()
    plot_x = []
    plot_y1 = []
    plot_y2 = []
    results = defaultdict()
    #
    # models = args.results_configs.split(",")
    #for model in models:
    #    with open(model) as f:
    #        model = model.split("/")[-1]
    #        model = model.replace(".json","")
    #        data = json.load(f)
    #        results[model] = defaultdict()
    #        results[model]["pred_score"] = mean(data["pred_score"])
    #        results[model]["avg_time_batch"] = data["avg_time_batch"]
    #        results[model]["totall_time"] = data["time"]
    #        plot_y1.append(results[model]["pred_score"])
    #        plot_y2.append(results[model]["totall_time"])
    #        plot_x.append(model)
    #json.dump(results, open(args.all_configs, "w"))
    #cifar10
    acc = [94.82, 95.55, 95.51, 91.48]
    time = [7,18, 12, 11]
    models = ["resnet18", "resnet50",  "little/big net", "skipnet-ffgate2"]
    fig, ax = plt.subplots()
    for i in range(len(acc)):
        ax.scatter(x=time[i], y=acc[i], label= models[i])
    ax.legend()
    #ride now this doesn't help 
    ax.invert_xaxis()
    #ax2 = ax1.twinx()
    #ax2 = ax2.scatter(x=plot_x,y=plot_y1, color="blue")
    
    
    #+fig.legend(models)
    plt.show()


    

if __name__ == '__main__':
    main()
