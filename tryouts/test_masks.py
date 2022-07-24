from torch import Tensor, zeros, rand
import torch
import pdb

def mask_down(t: Tensor, mask: Tensor) -> Tensor:
    # pdb.set_trace()
    return t[mask.bool()]

def mask_up(t: Tensor, mask: Tensor) -> Tensor:
    
    # pdb.set_trace()
    '''This method takes a downsized vector and upsizes it again, so that the new tensor
        has its values where the mask has its Ones.'''

    mask_as_list = list(mask)

    BS, C = len(mask_as_list), *(list(t.size())[1: ])
    small_batch_size = t.size()[0]
    output = zeros(BS, C)

    i = 0
    for j in range(BS):
        if mask_as_list[j]:
            output[j, :] = t[i, :]
            i += 1

    return output

mask = Tensor(([1, 0, 1, 0, 1])).long()
t = rand(5, 2)

t1 = mask_down(t, mask)

t2 = mask_up(t1, mask)

print(mask, t, t1, t2)
