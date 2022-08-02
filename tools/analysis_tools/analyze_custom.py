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
    models = ["resnet18", "resnet50", "Little/Big Net", "Branching Net","Multi-Scale Resolution Adaptive Net", "skipnet ffgate2", "CGNet - Skipping Channels"]

    flops = [0.56, 1.31 , 0.64, 0.8 , 0.013 , 0.94 , 0.015 ]
    flops = [1000 * x for x in flops]
    acc = [94.82, 95.55, 95.51, 78, 90.21, 91.48, 89.84]
    #imagenette
    #acc = [93.25,93.43, 93.99,  87, 90,81.91, 88.35, 89.66, 88.41]
    #flops  = [1.82, 4.12, 2.37, 5.5, 3.69, 10, 1.15,0.78, 1.9 ]
    #models = ["resnet18", "resnet50",  "little/big net", 
    #"skipnet-ffgate1", "skipnet-rnngate", "BrachingNet", "Multiscale-RANet", "CGNET","GRGB-Net"]
    fig, ax = plt.subplots()
    for i in range(len(acc)):
        ax.scatter(x=flops[i], y=acc[i], label= models[i])
    ax.legend()
    #ride now this doesn't help
    ax.set_ylabel('accuracy (%)') 
    ax.set_xlabel("computational cost (MFLOPs)")
    ax.invert_xaxis()
    #ax2 = ax1.twinx()
    #ax2 = ax2.scatter(x=plot_x,y=plot_y1, color="blue")
    
    
    #+fig.legend(models)
    plt.show()


    

if __name__ == '__main__':
    main()
