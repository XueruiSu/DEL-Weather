"""
@author: Xuerui Su
This file is the scripts to assist training
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.model_UNet import FNO3d, FNO3d_UQ, FNO3d_UQ_Res, UNet_O
from optim.Adam import Adam
from utils.utilities3 import count_params, LpLoss
from buffer import ReplayMemory
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_defination(modelConfig):
    modes = modelConfig['modes']
    width = modelConfig['width']
    learning_rate = modelConfig['learning_rate']
    scheduler_step = modelConfig['scheduler_step']
    scheduler_gamma = modelConfig['scheduler_gamma']
    model_s = modelConfig['model_s']
    
    # Data driven model
    model_datadriven = FNO3d(modes, modes, modes, width).cuda()
    print("Data driven model parameters number:", count_params(model_datadriven))
    optimizer_datadriven = Adam(model_datadriven.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler_datadriven = torch.optim.lr_scheduler.StepLR(optimizer_datadriven, step_size=scheduler_step, gamma=scheduler_gamma)

    # physics residual model
    model_physics = FNO3d(6, 6, 6, 20).cuda()
    print("physics residual model parameters number:", count_params(model_physics))
    optimizer_physics = Adam(model_physics.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler_physics = torch.optim.lr_scheduler.StepLR(optimizer_physics, step_size=scheduler_step, gamma=scheduler_gamma)

    if model_s == 'FNO':
        model_UQ = FNO3d_UQ(modes, modes, modes, width, T=modelConfig['T_in']+2*modelConfig['T']).cuda()
    elif model_s == 'Res':
        model_UQ = FNO3d_UQ_Res(modes, modes, modes, 32, T=modelConfig['T_in']+2*modelConfig['T']).cuda()
    elif model_s == 'UNet':
        model_UQ = UNet_O(in_ch=modelConfig['T_in']+2*modelConfig['T'], output_ch=modelConfig['T'], ch=16, ch_mult=[1,2,1,1], attn=True).cuda()
    print("confidence quantification model parameters number:", count_params(model_UQ))
    optimizer_UQ = Adam(model_UQ.parameters(), lr=8e-4, weight_decay=8e-4)
    scheduler_UQ = torch.optim.lr_scheduler.StepLR(optimizer_UQ, step_size=scheduler_step, gamma=scheduler_gamma)

    buffer = ReplayMemory(500, 0)
    myloss = LpLoss(size_average=False)

    model_dict_dd = {'model': model_datadriven, 'optimizer': optimizer_datadriven, 
                     'scheduler': scheduler_datadriven, 'myloss': myloss}
    model_dict_p = {'model': model_physics, 'optimizer': optimizer_physics, 
                     'scheduler': scheduler_physics, 'myloss': myloss}
    model_dict_UQ = {'model': model_UQ, 'optimizer': optimizer_UQ, 'scheduler': scheduler_UQ, 'myloss': myloss, 'buffer': buffer}
    return model_dict_dd, model_dict_p, model_dict_UQ

# FNO 3d train function
def training_3d(data_dict, model_dict, modelConfig, model_postprocesse):
    # train_laoder: (x, y), which need normalization
    # T: numbers of output frames, T_in: numbers of input frames
    # x: (B, H, W, T, T_in), y: (B, H, W, T)
    # data_dict = {'y_normalizer': y_normalizer, 'train_loader': train_loader}
    # model_dict = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler, 
    #     'model_postprocesse': model_postprocesse, 'myloss': myloss}
    model_dict['model'].train()
    train_mse = 0
    train_l2 = 0
    for x, y_solver, y in data_dict['train_loader']:
        x, y_solver, y = x.cuda(), y_solver.cuda(), y.cuda()
        
        model_dict['optimizer'].zero_grad()
        out, y_solver, y = model_postprocesse(model_dict, data_dict, x, y_solver, y, modelConfig, state='train')
        
        mse = F.mse_loss(out, y, reduction='mean')
        l2 = model_dict['myloss'](out.view(modelConfig['batch_size'], -1), y.view(modelConfig['batch_size'], -1))
        l2.backward()
        model_dict['optimizer'].step()
        
        train_mse += mse.item()
        train_l2 += l2.item()
    train_mse /= len(data_dict['train_loader'])
    train_l2 /= modelConfig['ntrain']
    model_dict['scheduler'].step()
    return train_mse, train_l2






# FNO 2d train function with autoregressive mode
def training_2d_ar(data_dict, model_dict, modelConfig):
    # train_laoder: (x, y), which do not need normalization
    # T: numbers of output frames, T_in: numbers of input frames
    # x: (B, H, W, T_in) as the input, y: (B, H, W, T) choose a frame as the output
    model_dict['model'].train()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in data_dict['train_loader']:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        for t in range(0, modelConfig['T'], modelConfig['step']):
            y = yy[..., t:t + modelConfig['step']]
            im = model_dict['model'](xx)
            loss += model_dict['myloss'](im.reshape(modelConfig['batch_size'], -1), y.reshape(modelConfig['batch_size'], -1))
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., modelConfig['step']:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = model_dict['myloss'](pred.reshape(modelConfig['batch_size'], -1), yy.reshape(modelConfig['batch_size'], -1))
        train_l2_full += l2_full.item()

        model_dict['optimizer'].zero_grad()
        loss.backward()
        model_dict['optimizer'].step()
        
    model_dict['scheduler'].step()
    return train_l2_step, train_l2_full

# FNO 2d train function without autoregressive mode
def training_2d(model, myloss, optimizer, scheduler, train_loader, y_normalizer, dict):
    # train_laoder: (x, y), which do not need normalization
    # T: numbers of output frames, T_in: numbers of input frames
    # x: (B, H, W, T, T_in), choose (B, H, W, t, T_in) as the input
    # y: (B, H, W, T) choose a frame as the output
    # from fourier_2d_tuned.py(myloss is 'hsloss' class)
    model.train()
    train_l2 = 0
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, dict['T']):
            x = xx[:,:,:,t,:]
            y = yy[:,:,:,t]

            out = model(x)
            loss = myloss(out.reshape(dict['batch_size'], dict['S'], dict['S']), 
                          y.reshape(dict['batch_size'], dict['S'], dict['S']))
            train_l2 += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    scheduler.step()
    return train_l2
