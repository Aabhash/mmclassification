from mmcls.models.backbones.dn_cgnet import CGResNet
from mmcls.models.backbones.multi_scale_dn import MultiScaleNet
from mmcls.models.backbones.multi_scale_dn_cifar import MultiScaleNetCifar

from tools.analysis_tools.flop_counter import FlopCounter
from torch import rand

from fvcore.nn import FlopCountAnalysis
import os
import json


def analyze_flops_msn(model, H, W, random_inp, exit_dir):
    print(" _______________________________________ \n")
    print("With flop_counter.py: ")
    if H == 224:
        print("Total Flops for MS-RANet on Imagenette")
    elif H == 32:
        print("Total flops for MS-RANet on Cifar10")

    cls_flops, _ = fc.measure_model(model, H, W)

    with open(os.path.join(exit_dir, 'exits.json')) as f:
        exit_counts = json.load(f).get("meta")

    if exit_counts:
        expected_flops = 0
        sample_size = sum(exit_counts.values())
        for i, exit in enumerate(exit_counts):
            prop = exit_counts[exit] * 1.0 / sample_size
            expected_flops += prop * cls_flops[i]

    flops = FlopCountAnalysis(model, random_inp)

    print("With FlopCountAnalysis:")
    print(" _______________________________________ \n")
    print(f"Total Flops: {flops.total() / 1e6:.4f}M,")
    print(f"Flops by Operator: {flops.by_operator()}")
    print(" _______________________________________ \n")

    rounder = lambda x: f'{x:,}'
    result = {
        'flop_by_exit': [rounder(c) for c in cls_flops],
        'expected_flops': rounder(expected_flops),
        'total_flop_FP': rounder(sum(cls_flops)),
        'last_flop_FCA': rounder(flops.total()),
        'flop_by_operator': flops.by_operator()
    }

    with open(os.path.join(exit_dir, 'flop_analysis.json'), "w+") as f:
        json.dump(result, f)


if __name__ == "__main__":

    fc = FlopCounter()
    CGN = CGResNet(
        18,
        [3, 3, 3],
        partitions=4,
        ginit=1
    )
    MSNC = MultiScaleNetCifar(
        growth_rate=4,
        channels=16,
        n_scales=3,
        n_blocks=2,
        step=4,
        bnwidth=[4, 2, 1],
    )
    MSN = MultiScaleNet(
        growth_rate=16,
        channels=32,
        n_scales=4,
        n_blocks=2,
        step=8,
        bnwidth=[4, 2, 2, 1],
        gr_factor=[1, 2, 2, 4],
    )

    H_Imagenette = 224
    W_Imagenette = 224

    H_Cifar = 32
    W_Cifar = 32

    random_Imagenette = rand(1, 3, H_Imagenette, W_Imagenette)
    random_Cifar = rand(1, 3, H_Cifar, W_Cifar)

    analyze_flops_msn(
        MSN,
        H_Imagenette,
        W_Imagenette,
        random_Imagenette,
        './results/multiscale-ranet-imagenet/'
    )
    
    analyze_flops_msn(
        MSNC,
        H_Cifar,
        W_Cifar,
        random_Cifar,
        './results/multiscale-ranet-cifar/'
    )

    print(" _______________________________________ \n")

    print("Total Flops for CGN on Imagenette")
    # print(measure_model(CGN, H_Imagenette, W_Imagenette))

    # print("CGN Imagenette")
    # flops = FlopCountAnalysis(CGN, random_Imagenette)
    # print(f"Total Flops: {flops.total() / 1e6:.4f}M,")
    # print(f"Flops by Operator:", flops.by_operator())

    print(" _______________________________________ \n")

    print("Total Flops for CGN on Cifar")
    # print(measure_model(CGN, H_Cifar, W_Cifar))

    # print("CGN Cifar")
    # flops = FlopCountAnalysis(CGN, random_Cifar)
    # print(f"Total Flops: {flops.total() / 1e6:.4f}M,")
    # print(f"Flops by Operator:", flops.by_operator())

    print(" _______________________________________ \n")
