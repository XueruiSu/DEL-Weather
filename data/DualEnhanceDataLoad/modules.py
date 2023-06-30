import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import UnitGaussianNormalizer, count_params, LpLoss

def load_data_from_file(modelConfig):
    # Main model data (B, H, W, 40):
    train_u = torch.from_numpy(np.load(modelConfig['training_file_path'])['utn'])[:modelConfig['ntrain']]
    train_u_solver = torch.from_numpy(np.load(modelConfig['training_file_path'])['utn_solver'])[:modelConfig['ntrain']]
    test_u = torch.from_numpy(np.load(modelConfig['testing_file_path'])['utn'])[:modelConfig['ntest']]
    test_u_solver = torch.from_numpy(np.load(modelConfig['testing_file_path'])['utn_solver'])[:modelConfig['ntest']]
    return train_u, train_u_solver, test_u, test_u_solver

def reshape_slide_window(modelConfig, u):
    # u: (B, H, W, 40)
    # u_t, u_t_1: (B, H, W, T_in), (B, H, W, T)
    T_in = modelConfig['T_in']
    T = modelConfig['T']
    u_H = u.shape[0]
    u_W = u.shape[-1]-T_in-T+1
    u_t_0 = torch.zeros((u_H*u_W, u.shape[1], u.shape[2], T_in))
    u_t_1 = torch.zeros((u_H*u_W, u.shape[1], u.shape[2], T))
    for u_num_index in range(u_H):
        for u_index in range(u_W):
            u_t_0[u_W*u_num_index+u_index] = u[u_num_index, :, :, u_index:u_index+T_in]
            u_t_1[u_W*u_num_index+u_index] = u[u_num_index, :, :, u_index+T_in:u_index+T_in+T]  
    return u_t_0, u_t_1

def data_load(modelConfig, model_kind='datadriven'):
    if model_kind == 'datadriven':
        batch_str = 'batch_size_dd'
    elif model_kind == 'physics':
        batch_str = 'batch_size_p'
    elif model_kind == 'UQ_train':
        batch_str = 'batch_size_UQ_train'
    t1 = default_timer()
    # (B, H, W, 40)
    train_data, train_data_solver, test_data, test_data_solver = load_data_from_file(modelConfig)
    trajectory_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_data_solver), 
                                                    batch_size=modelConfig[batch_str], shuffle=False)
    train_a, train_u= reshape_slide_window(modelConfig, train_data)
    train_a_solver, train_u_solver= reshape_slide_window(modelConfig, train_data_solver)
    test_a, test_u= reshape_slide_window(modelConfig, test_data)
    test_a_solver, test_u_solver= reshape_slide_window(modelConfig, test_data_solver)
    assert (modelConfig['S'] == train_u.shape[-2])
    assert (modelConfig['T'] == train_u.shape[-1])
    # normalizer
    # (B, H, W, T_in)
    a_normalizer_cpu = UnitGaussianNormalizer(train_a)
    a_normalizer_cuda = UnitGaussianNormalizer(train_a)
    a_normalizer_cpu.cpu()
    a_normalizer_cuda.cuda()
    # (B, H, W, T)
    y_normalizer_cpu = UnitGaussianNormalizer(train_u)
    y_normalizer_cuda = UnitGaussianNormalizer(train_u)
    y_normalizer_cpu.cpu()
    y_normalizer_cuda.cuda()
    # normalization
    train_a_n = a_normalizer_cpu.encode(train_a)
    test_a_n = a_normalizer_cpu.encode(test_a)
    train_u_n = y_normalizer_cpu.encode(train_u)
    test_u_n = y_normalizer_cpu.encode(test_u)
    train_u_solver_n = y_normalizer_cpu.encode(train_u_solver)
    test_u_solver_n = y_normalizer_cpu.encode(test_u_solver)
    # dataloader:
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a_n, train_u_solver_n, train_u_n), 
                                                    batch_size=modelConfig[batch_str], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a_n, test_u_solver_n, test_u_n), 
                                                    batch_size=modelConfig[batch_str], shuffle=False)
    t2 = default_timer()
    data_dict = {
        'a_normalizer_cpu': a_normalizer_cpu,
        'a_normalizer_cuda': a_normalizer_cuda,
        'y_normalizer_cpu': y_normalizer_cpu,
        'y_normalizer_cuda': y_normalizer_cuda,
        'train_loader': train_loader, 
        'test_loader': test_loader,
        'trajectory_data_loader': trajectory_data_loader
    }
    print('data_loader preprocessing finished, time used:', t2-t1)
    return data_dict

