"""
@author: Xuerui Su
This file is the scripts to assist testing
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utilities3 import *
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from optim.Adam import Adam

torch.manual_seed(0)
np.random.seed(0)
 

# FNO 3d train function
def testing_3d(data_dict, model_dict, modelConfig, model_postprocesse):
    # train_laoder: (x, y), which need normalization
    # T: numbers of output frames, T_in: numbers of input frames
    # x: (B, H, W, T, T_in), y: (B, H, W, T)
    model_dict['model'].eval()
    first_re = 0
    with torch.no_grad():
        for x, y_solver, y in data_dict['test_loader']:
            x, y_solver, y = x.cuda(), y_solver.cuda(), y.cuda()
            out, y_solver, y, loss_list = model_postprocesse(model_dict, data_dict, x, y_solver, y, modelConfig, state='test')
            if first_re == 0:
                first_re = 1
                test_l2 = []
                for loss_list_index in range(len(loss_list)):
                    test_l2.append(loss_list[loss_list_index])
            else:
                for loss_list_index in range(len(loss_list)):
                    test_l2[loss_list_index] += (loss_list[loss_list_index])
    for test_l2_index in range(len(test_l2)):
        test_l2[test_l2_index] /= model_dict['ntest']
    return test_l2


# FNO 2d train function with autoregressive mode
def testing_2d(model, test_loader, y_normalizer, dict):
    # train_laoder: (x, y), which do not need normalization
    # T: numbers of output frames, T_in: numbers of input frames
    # x: (B, H, W, T_in) as the input, y: (B, H, W, T) choose a frame as the output
    lploss = LpLoss(size_average=False)
    
    model.eval()
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, dict['T'], dict['step']):
                y = yy[..., t:t + dict['step']]
                im = model(xx)
                
                loss += lploss(im.reshape(dict['batch_size'], -1), y.reshape(dict['batch_size'], -1))
                
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., dict['step']:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += lploss(pred.reshape(dict['batch_size'], -1), yy.reshape(dict['batch_size'], -1)).item()
    return test_l2_step, test_l2_full


# FNO 2d train function without autoregressive mode
def testing_2d(model, test_loader, y_normalizer, dict):
    # train_laoder: (x, y), which do not need normalization
    # T: numbers of output frames, T_in: numbers of input frames
    # x: (B, H, W, T, T_in), choose (B, H, W, t, T_in) as the input
    # y: (B, H, W, T) choose a frame as the output
    # from fourier_2d_tuned.py(myloss is 'hsloss' class)
    loss_k = 1
    loss_group = False
    lploss = LpLoss(size_average=False)
    hsloss = HsLoss(k=loss_k, group=loss_group, size_average=False)

    model.eval()
    test_l2 = 0
    test_l2_hp = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, dict['T']):
                x = xx[:, :, :, t, :]
                y = yy[:, :, :, t]

                out = model(x)
                test_l2 += lploss(out.reshape(dict['batch_size'], dict['S'], dict['S']), 
                                  y.reshape(dict['batch_size'], dict['S'], dict['S'])).item()
                test_l2_hp += hsloss(out.reshape(dict['batch_size'], dict['S'], dict['S']), 
                                     y.reshape(dict['batch_size'], dict['S'], dict['S'])).item()




