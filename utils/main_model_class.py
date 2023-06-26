import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utils.utilities3 import UnitGaussianNormalizer, count_params, LpLoss
from utils.scripts_calc_acc import calc_acc, Ranking_loss2
from utils.buffer import ReplayMemory
from model.FNO_2d import FNO2d, FNO2d_Physics, FNO2d_UQ
from model.UQ_sub_model import FNO2d_UQ_sample
from utils.Adam import Adam
from utils.NS_Solver_vorticity import solver_ns

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

class datadriven_model():
    def __init__(self, modelConfig):
        self.modelConfig = modelConfig
        # load training and test data
        # data_dict = {
        # 'a_normalizer_cpu': a_normalizer_cpu,
        # 'a_normalizer_cuda': a_normalizer_cuda,
        # 'y_normalizer_cpu': y_normalizer_cpu,
        # 'y_normalizer_cuda': y_normalizer_cuda,
        # 'train_loader': train_loader, 
        # 'test_loader': test_loader
        # }
        self.data_dict = data_load(modelConfig, model_kind='datadriven')
        
        # Data driven model
        modes = modelConfig['modes']
        width = modelConfig['width']
        learning_rate = modelConfig['learning_rate_dd']
        scheduler_step = modelConfig['scheduler_step_dd']
        scheduler_gamma = modelConfig['scheduler_gamma_dd']
        self.model = FNO2d(modes, modes, width, T_in=modelConfig['T_in']).cuda()
        print("Data driven model parameters number:", count_params(self.model))
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        self.myloss = LpLoss(size_average=False)
        self.S = self.modelConfig['S']
        self.T = self.modelConfig['T']        
        self.T_in = self.modelConfig['T_in']        
        
    def train_batch(self, x, y):
        self.model.train()
        x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        out = self.model(x).view(y.shape[0], self.S, self.S, self.T)
        out = self.data_dict['y_normalizer_cuda'].decode(out)
        y = self.data_dict['y_normalizer_cuda'].decode(y)
        mse = F.mse_loss(out, y, reduction='mean')
        l2 = self.myloss(out.view(y.shape[0], -1), y.view(y.shape[0], -1))
        l2.backward()
        self.optimizer.step()
        return mse.item()/x.shape[0], l2.item()/x.shape[0]
    
    def train_multi_batch(self, x, y, batch_num=2):
        mse_dict = []
        l2_dict = []
        for batch_index in range(batch_num):
            mse, l2 = self.train_batch(x, y)
            print("Batch num:", batch_index, "Train MSE Loss:", mse, "Train L2 Loss:", l2)
            mse_dict.append(mse_dict)
            l2_dict.append(l2)
        return mse_dict, l2_dict
    
    def train_epoch(self, ):
        # train_laoder: (x, y), which need normalization
        # T: numbers of output frames, T_in: numbers of input frames
        # x: (B, H, W, T, T_in), y: (B, H, W, T)
        self.model.train()
        train_mse, train_l2 = 0, 0
        for x, y_solver, y in self.data_dict['train_loader']:
            mse, l2 = self.train_batch(x, y)
            train_mse += mse*x.shape[0]
            train_l2 += l2*x.shape[0]
        train_mse /= len(self.data_dict['train_loader'])
        train_l2 /= (len(self.data_dict['train_loader'])*y.shape[0])
        self.scheduler.step()
        return train_mse, train_l2
    
    def test(self, ):
        # test_loader: (x, y), which need normalization
        # T: numbers of output frames, T_in: numbers of input frames
        # x: (B, H, W, T, T_in), y: (B, H, W, T)
        test_mse, test_l2 = 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            for x, y_solver, y in self.data_dict['test_loader']:
                x, y_solver, y = x.cuda(), y_solver.cuda(), y.cuda()
                out = self.model(x).view(y.shape[0], self.S, self.S, self.T)
                out = self.data_dict['y_normalizer_cuda'].decode(out)
                y = self.data_dict['y_normalizer_cuda'].decode(y)
                mse = F.mse_loss(out, y, reduction='mean')
                l2 = self.myloss(out.view(y.shape[0], -1), y.view(y.shape[0], -1))
                test_mse += mse.item()
                test_l2 += l2.item()
        test_mse /= len(self.data_dict['test_loader'])
        test_l2 /= (len(self.data_dict['test_loader'])*y.shape[0])
        return test_mse, test_l2

    def multi_pre(self, step=2):
        test_mse_step = []
        test_l2_step = []
        multi_first_mask = 0
        self.model.eval()
        with torch.no_grad():
            for test_data, test_data_solver in self.data_dict['trajectory_data_loader']:
                # without normalization
                xx, yy = test_data[..., :self.T_in].cuda(), test_data[..., self.T_in:].cuda()
                xx = self.data_dict['a_normalizer_cuda'].encode(xx)
                test_mse_batch, test_l2_batch = [], []
                for t in range(step):
                    y = yy[..., t:t+1]
                    im = self.model(xx).view(y.shape[0], self.S, self.S, self.T)
                    im = self.data_dict['y_normalizer_cuda'].decode(im)
                    loss = self.myloss(im.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))/y.shape[0]
                    mse = F.mse_loss(im, y, reduction='mean')
                    test_mse_batch.append(mse.item())
                    test_l2_batch.append(loss.item())
                    xx = self.data_dict['a_normalizer_cuda'].decode(xx)
                    xx = torch.cat((xx[..., 1:], im), dim=-1)
                    xx = self.data_dict['a_normalizer_cuda'].encode(xx)
                test_mse_batch, test_l2_batch = np.array(test_mse_batch), np.array(test_l2_batch)
                if multi_first_mask == 0:
                    multi_first_mask = 1
                    test_mse_step, test_l2_step = test_mse_batch, test_l2_batch
                else:
                    test_mse_step += test_mse_batch
                    test_l2_step += test_l2_batch
        return test_mse_step/len(self.data_dict['trajectory_data_loader']), test_l2_step/len(self.data_dict['trajectory_data_loader']) 
    
    def multi_pre_plot(self, step=10, epoch=1):
        test_mse_step, test_l2_step = self.multi_pre(step)
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(test_mse_step, label='Multi Step MSE Error')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('MSE Error')
        plt.subplot(1,2,2)
        plt.plot(test_l2_step, label='Multi Step Relative Error')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('Relative Error')
        plt.savefig(f"{self.modelConfig['MODEL_PATH']}/figure/{epoch}_multi_step_dd.png")
        return test_mse_step, test_l2_step
    
class physics_model():
    def __init__(self, modelConfig):
        self.modelConfig = modelConfig
        # load training and test data
        # data_dict = {
        # 'a_normalizer_cpu': a_normalizer_cpu,
        # 'a_normalizer_cuda': a_normalizer_cuda,
        # 'y_normalizer_cpu': y_normalizer_cpu,
        # 'y_normalizer_cuda': y_normalizer_cuda,
        # 'train_loader': train_loader, 
        # 'test_loader': test_loader
        # }
        self.data_dict = data_load(modelConfig, model_kind='physics')
        
        # physics residual model
        modes = 6
        width = 20
        learning_rate = modelConfig['learning_rate_p']
        scheduler_step = modelConfig['scheduler_step_p']
        scheduler_gamma = modelConfig['scheduler_gamma_p']
        
        self.model = FNO2d_Physics(modes, modes, width, T_in=modelConfig['T_in']+modelConfig['T']).cuda()
        print("physics residual model parameters number:", count_params(self.model))
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        self.myloss = LpLoss(size_average=False)
        self.S = self.modelConfig['S']
        self.T = self.modelConfig['T']
        self.T_in = self.modelConfig['T_in']
        
    def train_batch(self, x, y_solver, y):
        self.model.train()
        x, y_solver, y = x.cuda(), y_solver.cuda(), y.cuda()
        
        self.optimizer.zero_grad()
        out = self.model(x, y_solver).view(y.shape[0], self.S, self.S, self.T)
        out = out.view(y.shape[0], self.S, self.S, self.T) + y_solver.view(y.shape[0], self.S, self.S, self.T)
        
        y = self.data_dict['y_normalizer_cuda'].decode(y)
        y_solver = self.data_dict['y_normalizer_cuda'].decode(y_solver)
        out = self.data_dict['y_normalizer_cuda'].decode(out)
        
        mse = F.mse_loss(out, y, reduction='mean')
        l2 = self.myloss(out.view(y.shape[0], -1), y.view(y.shape[0], -1))
        l2.backward()
        self.optimizer.step()
        
        mse_solver = F.mse_loss(y_solver, y, reduction='mean')
        l2_solver = self.myloss(y_solver.view(y.shape[0], -1), y.view(y.shape[0], -1))
        return mse.item()/x.shape[0], l2.item()/x.shape[0], mse_solver.item()/x.shape[0], l2_solver.item()/x.shape[0]
    
    def train_multi_batch(self, x, y_solver, y, batch_num=2):
        mse_dict, l2_dict = [], []
        mse_solver_dict, l2_solver_dict = [], []
        for batch_index in range(batch_num):
            mse, l2, mse_solver, l2_solver = self.train_batch(x, y_solver, y)
            print("Batch num:", batch_index, "Train MSE Loss:", mse, "Train L2 Loss:", l2)
            print("Batch num:", batch_index, "Train MSE Solver Loss:", mse_solver, "Train L2 Solver Loss:", l2_solver)
            mse_dict.append(mse_dict)
            l2_dict.append(l2)
            mse_solver_dict.append(mse_solver)
            l2_solver_dict.append(l2_solver)
        return mse_dict, l2_dict, mse_solver_dict, l2_solver_dict
    
    def train_epoch(self, ):
        # train_laoder: (x, y), which need normalization
        # T: numbers of output frames, T_in: numbers of input frames
        # x: (B, H, W, T, T_in), y: (B, H, W, T)
        self.model.train()
        train_mse, train_l2, train_mse_s, train_l2_s = 0, 0, 0, 0
        for x, y_solver, y in self.data_dict['train_loader']:
            mse, l2, mse_solver, l2_solver = self.train_batch(x, y_solver, y)
            train_mse += mse*x.shape[0]
            train_l2 += l2*x.shape[0]
            train_mse_s += mse_solver*x.shape[0]
            train_l2_s += l2_solver*x.shape[0]
        train_mse /= len(self.data_dict['train_loader'])
        train_l2 /= (len(self.data_dict['train_loader'])*y.shape[0])
        train_mse_s /= len(self.data_dict['train_loader'])
        train_l2_s /= (len(self.data_dict['train_loader'])*y.shape[0])
        self.scheduler.step()
        return train_mse, train_l2, train_mse_s, train_l2_s
    
    def test(self, ):
        # test_loader: (x, y), which need normalization
        # T: numbers of output frames, T_in: numbers of input frames
        # x: (B, H, W, T, T_in), y: (B, H, W, T)
        test_mse, test_l2, test_mse_solver, test_l2_solver = 0.0, 0.0, 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            for x, y_solver, y in self.data_dict['test_loader']:
                x, y_solver, y = x.cuda(), y_solver.cuda(), y.cuda()
                
                out = self.model(x, y_solver).view(y.shape[0], self.S, self.S, self.T)
                out = out.view(y.shape[0], self.S, self.S, self.T) + y_solver.view(y.shape[0], self.S, self.S, self.T)
                
                out = self.data_dict['y_normalizer_cuda'].decode(out)
                y_solver = self.data_dict['y_normalizer_cuda'].decode(y_solver)
                y = self.data_dict['y_normalizer_cuda'].decode(y)
                
                mse = F.mse_loss(out, y, reduction='mean')
                l2 = self.myloss(out.view(y.shape[0], -1), y.view(y.shape[0], -1))
                mse_solver = F.mse_loss(y_solver, y, reduction='mean')
                l2_solver = self.myloss(y_solver.view(y.shape[0], -1), y.view(y.shape[0], -1))
                                
                test_mse += mse.item()
                test_l2 += l2.item()
                test_mse_solver += mse_solver.item()
                test_l2_solver += l2_solver.item()
        test_mse /= len(self.data_dict['test_loader'])
        test_l2 /= (len(self.data_dict['test_loader'])*y.shape[0])
        test_mse_solver /= len(self.data_dict['test_loader'])
        test_l2_solver /= (len(self.data_dict['test_loader'])*y.shape[0])
        return test_mse, test_l2, test_mse_solver, test_l2_solver

    def multi_pre(self, step=10):
        test_mse_step, test_l2_step = [], []
        multi_first_mask = 0
        self.model.eval()
        with torch.no_grad():
            for test_data, test_data_solver in self.data_dict['trajectory_data_loader']:
                # without normalization
                xx, yy = test_data[..., :self.T_in].cuda(), test_data[..., self.T_in:].cuda()
                yy_solver = test_data_solver[..., self.T_in:].cuda()
                xx = self.data_dict['a_normalizer_cuda'].encode(xx)
                test_mse_batch, test_l2_batch = [], []
                for t in range(step):
                    y, y_solver = yy[..., t:t+1], yy_solver[..., t:t+1]
                    y_solver = self.data_dict['y_normalizer_cuda'].encode(y_solver).view(y.shape[0], self.S, self.S, self.T)
                    im = self.model(xx, y_solver).view(y.shape[0], self.S, self.S, self.T) + y_solver
                    im = self.data_dict['y_normalizer_cuda'].decode(im)
                    loss = self.myloss(im.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))/y.shape[0]
                    mse = F.mse_loss(im, y, reduction='mean')
                    test_mse_batch.append(mse.item())
                    test_l2_batch.append(loss.item())
                    xx = self.data_dict['a_normalizer_cuda'].decode(xx)
                    xx = torch.cat((xx[..., 1:], im), dim=-1)
                    xx = self.data_dict['a_normalizer_cuda'].encode(xx)
                test_mse_batch, test_l2_batch = np.array(test_mse_batch), np.array(test_l2_batch)
                if multi_first_mask == 0:
                    multi_first_mask = 1
                    test_mse_step, test_l2_step = test_mse_batch, test_l2_batch
                else:
                    test_mse_step += test_mse_batch
                    test_l2_step += test_l2_batch
        return test_mse_step/len(self.data_dict['trajectory_data_loader']), test_l2_step/len(self.data_dict['trajectory_data_loader']) 

    def multi_pre_plot(self, step=10, epoch=1):
        test_mse_step, test_l2_step = self.multi_pre(step)
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(test_mse_step, label='Multi Step MSE Error')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('MSE Error')
        plt.subplot(1,2,2)
        plt.plot(test_l2_step, label='Multi Step Relative Error')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('Relative Error')
        plt.savefig(f"{self.modelConfig['MODEL_PATH']}/figure/{epoch}_multi_step_p.png")
        plt.show()
        plt.close()
        return test_mse_step, test_l2_step