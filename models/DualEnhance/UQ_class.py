import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.utilities3 import count_params, LpLoss
from utils.scripts_calc_acc import Ranking_loss2, Ranking_loss_subnet
from utils.buffer import ReplayMemory
from modules.FNO_2d import FNO2d_UQ_mean
from modules.UQ_sub_model import FNO2d_UQ_sample, FNO2d_UQ_sample_mainmodel, UQ_sample_mainmodel, FNO2d_UQ_NLL, FNO2d_UQ_Ensemble
from optim.Adam import Adam     
from data.DualEnhanceDataLoad.modules import data_load
from configs.Climax_train_modelparam import * # hyperparameters
from data.seqrecord.module import MultiSourceDataModule

class UQ_model():
    def __init__(self, modelConfig):
        self.modelConfig = modelConfig
        self.datamodule_class = MultiSourceDataModule(dict_root_dirs, dict_data_spatial_shapes, 
                                                dict_single_vars, dict_atmos_vars, dict_hrs_each_step, 
                                                dict_max_predict_range, batch_size, dict_metadata_dirs, 
                                                shuffle_buffer_size=shuffle_buffer_size, 
                                                val_shuffle_buffer_size=val_shuffle_buffer_size, 
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                use_old_loader=use_old_loader)
        
        modes = modelConfig['modes_UQ']
        width = modelConfig['width_UQ']
        learning_rate = modelConfig['learning_rate_UQ']
        scheduler_step = modelConfig['scheduler_step_UQ']
        scheduler_gamma = modelConfig['scheduler_gamma_UQ']
        self.model_s = modelConfig['model_s']
        if self.model_s == 'FNO':
            self.model = FNO2d_UQ_mean(modes, modes, width, T_in=modelConfig['T_in'], T=modelConfig['T']).cuda()
        print("confidence quantification model parameters number:", count_params(self.model))
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=8e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        self.buffer = ReplayMemory(500, 0)
        self.myloss = LpLoss(size_average=False)
        self.S = self.modelConfig['S']
        self.T = self.modelConfig['T']
        self.T_in = self.modelConfig['T_in']
        self.batch_size_UQ_train = self.modelConfig['batch_size_UQ_train']
        self.batch_size_dual = self.modelConfig['batch_size_dual']
        
    def buffer_push(self, dd_model, p_model):
        # save (ut,utn_hat, utn)
        dd_model.model.eval()
        p_model.model.eval()
        with torch.no_grad():
            for x, y_solver, y in self.data_dict['train_loader']:
                x, out_solver, y = x.cuda(), y_solver.cuda(), y.cuda()
                out_d = dd_model.model(x).view(y.shape[0], self.S, self.S, self.T)
                out_p = p_model.model(x, out_solver).view(y.shape[0], self.S, self.S, self.T) + out_solver.view(y.shape[0], self.S, self.S, self.T)
                # the data here is after normalization
                for buffer_index in range(x.shape[0]):
                    self.buffer.push(x[buffer_index:buffer_index+1].detach().cpu().numpy(), 
                                     out_d[buffer_index:buffer_index+1].detach().cpu().numpy(), 
                                     out_p[buffer_index:buffer_index+1].detach().cpu().numpy(), 
                                     y[buffer_index:buffer_index+1].detach().cpu().numpy())
        
    def sample_batch_tensor(self):
        # load training data from buffer  
        ut0_n_UQ, utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ = self.buffer.sample(self.batch_size_UQ_train)
        ut0_n_UQ = torch.tensor(ut0_n_UQ.reshape(-1, self.S, self.S, self.T_in))[:self.batch_size_UQ_train]
        utn0_d_n_UQ = torch.tensor(utn0_d_n_UQ.reshape(-1, self.S, self.S, self.T))[:self.batch_size_UQ_train]
        utn0_p_n_UQ = torch.tensor(utn0_p_n_UQ.reshape(-1, self.S, self.S, self.T))[:self.batch_size_UQ_train]
        utn0_n_UQ = torch.tensor(utn0_n_UQ.reshape(-1, self.S, self.S, self.T))[:self.batch_size_UQ_train]
        return ut0_n_UQ, utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ
    
    def loss_UQ_mean(self, R_1_pre, R_2_pre, R_1, R_2):
        # out: (B, H, W, 2*T), R_1/R_2: (B, H, W, T)
        mse_loss = (F.mse_loss(R_1_pre, R_1, reduction='mean') + F.mse_loss(R_2_pre, R_2, reduction='mean')) / 2
        l2_loss = (self.myloss(R_1_pre, R_1) + self.myloss(R_2_pre, R_2)) / (2*R_1.shape[0])
        rank_loss = Ranking_loss2(R_1_pre, R_2_pre, R_1, R_2)
        mixed_loss = (self.modelConfig['UQ_mse']*mse_loss + self.modelConfig['UQ_l2']*l2_loss + self.modelConfig['UQ_Rank']*rank_loss)
        return mixed_loss, mse_loss, l2_loss, rank_loss    
    
    def loss_UQ_subnet(self, R_1_pre, R_2_pre, R_1, R_2):
        # R_1_pre/R_2_pre: (B, 1), R_1/R_2: (B, H, W, T)
        for _ in range(3):
            R_1, R_2 = torch.mean(R_1, dim=-1), torch.mean(R_2, dim=-1)
        mse_loss = ((R_1_pre - R_1)**2 + (R_2_pre - R_2)**2).sum() / (2*R_1.shape[0])
        l2_loss = (self.myloss(R_1_pre, R_1) + self.myloss(R_2_pre, R_2)) / (2*R_1.shape[0])
        rank_loss = Ranking_loss_subnet(R_1_pre, R_2_pre, R_1, R_2)
        mixed_loss = (self.modelConfig['UQ_mse']*mse_loss + self.modelConfig['UQ_l2']*l2_loss + self.modelConfig['UQ_Rank']*rank_loss)
        return mixed_loss, mse_loss, l2_loss, rank_loss    
    
    def loss_UQ_subnet_RE(self, R_1_pre, R_2_pre, R_1, R_2):
        # R_1_pre/R_2_pre: (B, 1), R_1/R_2: (B, 1)
        mse_loss = ((R_1_pre - R_1)**2 + (R_2_pre - R_2)**2).sum() / (2*R_1.shape[0])
        l2_loss = (self.myloss(R_1_pre, R_1) + self.myloss(R_2_pre, R_2)) / (2*R_1.shape[0])
        rank_loss = Ranking_loss_subnet(R_1_pre, R_2_pre, R_1, R_2)
        mixed_loss = (self.modelConfig['UQ_mse']*mse_loss + self.modelConfig['UQ_l2']*l2_loss + self.modelConfig['UQ_Rank']*rank_loss)
        return mixed_loss, mse_loss, l2_loss, rank_loss    
    
    def NLL_UQ(self, var_d, var_p, R_1, R_2):
        # var_d/var_p: (B, H, W, T), R_1/R_2: (B, 1)
        for _ in range(3):
            var_d, var_p = torch.mean(var_d, dim=-1), torch.mean(var_p, dim=-1)
        mse_loss = ((var_d - R_1)**2 + (var_p - R_2)**2).sum() / (2*R_1.shape[0])
        l2_loss = (self.myloss(var_d, R_1) + self.myloss(var_p, R_2)) / (2*R_1.shape[0])
        rank_loss = Ranking_loss_subnet(var_d, var_p, R_1, R_2)
        return mse_loss, l2_loss, rank_loss    
    
    def loss_UQ_NLL(self, log_var_d, log_var_p, utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ, R_1, R_2):
        # log_var_d/log_var_p: (B, H, W, T), R_1/R_2: (B, 1)
        out_d = self.data_dict['y_normalizer_cuda'].decode(utn0_d_n_UQ)
        out_p = self.data_dict['y_normalizer_cuda'].decode(utn0_p_n_UQ)
        out = self.data_dict['y_normalizer_cuda'].decode(utn0_n_UQ)
        var_d = torch.exp(log_var_d)
        NLL_d = 0.5 * (log_var_d + (out - out_d)**2 / var_d)
        var_p = torch.exp(log_var_p)
        NLL_p = 0.5 * (log_var_p + (out - out_p)**2 / var_p)
        NLL = (NLL_d.mean() + NLL_p.mean()) / 2
        mse_loss, l2_loss, rank_loss = self.NLL_UQ(var_d, var_p, R_1, R_2)
        return NLL, mse_loss, l2_loss, rank_loss    
    
    def decode_all(self, utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ):
        out_d = self.data_dict['y_normalizer_cuda'].decode(utn0_d_n_UQ)
        out_p = self.data_dict['y_normalizer_cuda'].decode(utn0_p_n_UQ)
        out = self.data_dict['y_normalizer_cuda'].decode(utn0_n_UQ)
        R_1 = self.modelConfig['dual_judge']*torch.abs(out - out_d).cuda()
        R_2 = torch.abs(out - out_p).cuda()
        return R_1, R_2

    def rel_RE(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        return diff_norms/y_norms

    def decode_all_RE(self, utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ):
        out_d = self.data_dict['y_normalizer_cuda'].decode(utn0_d_n_UQ)
        out_p = self.data_dict['y_normalizer_cuda'].decode(utn0_p_n_UQ)
        out = self.data_dict['y_normalizer_cuda'].decode(utn0_n_UQ)
        R1_relative = self.rel_RE(out_d.reshape(out_d.shape[0], -1), out.reshape(out.shape[0], -1)).cuda()
        R2_relative = self.rel_RE(out_p.reshape(out_p.shape[0], -1), out.reshape(out.shape[0], -1)).cuda()
        # R1_relative/R2_relative: (B, 1)
        return R1_relative, R2_relative    

    def Loss_Ensemble(self, mean_d, mean_p, utn0_n_UQ, var_d, var_p, R_1, R_2):
        # var_d/var_p: (B, H, W, T), R_1/R_2: (B, H, W, T)
        for _ in range(3):
            var_d, var_p = torch.mean(var_d, dim=-1), torch.mean(var_p, dim=-1)
        _, mse_loss, l2_loss, rank_loss = self.loss_UQ_subnet(var_d, var_p, R_1, R_2)
        out_d = self.data_dict['y_normalizer_cuda'].decode(mean_d)
        out_p = self.data_dict['y_normalizer_cuda'].decode(mean_p)
        out = self.data_dict['y_normalizer_cuda'].decode(utn0_n_UQ)
        Loss_d, Loss_p = self.myloss(out_d, out), self.myloss(out_p, out)
        
        return (Loss_d + Loss_p) / (2*out.shape[0]), mse_loss, l2_loss, rank_loss
    
    def train_batch(self, ut0_n_UQ, utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ):
        self.model.train()
        x = torch.concat([ut0_n_UQ, utn0_d_n_UQ, utn0_p_n_UQ], dim=3).cuda()
        R_1, R_2 = self.decode_all(utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ)
        self.optimizer.zero_grad()
        if self.model_s == 'FNO':
            R_1_pre, R_2_pre = self.model(x)
            R_1_pre, R_2_pre = self.data_dict['y_normalizer_cuda'].decode(R_1_pre), self.data_dict['y_normalizer_cuda'].decode(R_2_pre)
            mixed_loss, mse, l2, rank_loss = self.loss_UQ_mean(R_1_pre, R_2_pre, R_1, R_2)
        mixed_loss.backward()
        self.optimizer.step()
        return mse.item(), l2.item(), rank_loss.item()
    
    def train_epoch(self, dd_model, p_model):
        self.buffer_push(dd_model, p_model)
        train_mse = 0
        train_l2 = 0
        train_rank_loss = 0
        for _ in range(self.modelConfig['UQ_repeat']):
            ut0_n_UQ, utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ = self.sample_batch_tensor()
            mse, l2_loss, rank_loss = self.train_batch(ut0_n_UQ.cuda(), utn0_d_n_UQ.cuda(), utn0_p_n_UQ.cuda(), utn0_n_UQ.cuda())
            train_mse += mse
            train_l2 += l2_loss
            train_rank_loss += rank_loss
        train_mse /= self.modelConfig['UQ_repeat']
        train_l2 /= self.modelConfig['UQ_repeat']
        train_rank_loss /= self.modelConfig['UQ_repeat']
        self.scheduler.step()
        return train_mse, train_l2, train_rank_loss
    
    def test_batch(self, ut0_n_UQ, utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ):
        self.model.eval()
        x = torch.concat([ut0_n_UQ, utn0_d_n_UQ, utn0_p_n_UQ], dim=3).cuda()
        R_1, R_2 = self.decode_all(utn0_d_n_UQ, utn0_p_n_UQ, utn0_n_UQ)
        if self.model_s == 'FNO':
            R_1_pre, R_2_pre = self.model(x)
            R_1_pre, R_2_pre = self.data_dict['y_normalizer_cuda'].decode(R_1_pre), self.data_dict['y_normalizer_cuda'].decode(R_2_pre)
            _, test_mse, test_l2, test_rank_loss = self.loss_UQ_mean(R_1_pre, R_2_pre, R_1, R_2)
        return test_mse.item(), test_l2.item(), test_rank_loss.item()
    
    def test(self, dd_model, p_model):
        dd_model.model.eval()
        p_model.model.eval()
        test_mse = 0
        test_l2 = 0
        test_rank_loss = 0
        with torch.no_grad():
            for x, y_solver, y in self.data_dict['test_loader']:
                x, y_solver, y = x.cuda(), y_solver.cuda(), y.cuda()
                out_d = dd_model.model(x).view(y.shape[0], self.S, self.S, self.T)
                out_p = p_model.model(x, y_solver).view(y.shape[0], self.S, self.S, self.T) + y_solver.view(y.shape[0], self.S, self.S, self.T)
                mse, l2_loss, rank_loss = self.test_batch(x, out_d, out_p, y)
                test_mse += mse
                test_l2 += l2_loss
                test_rank_loss += rank_loss
        test_mse /= len(self.data_dict['test_loader'])
        test_l2 /= len(self.data_dict['test_loader'])
        test_rank_loss /= len(self.data_dict['test_loader'])
        return test_mse, test_l2, test_rank_loss
    
    def UQ_acc_test(self, dd_model, p_model):
        dd_model.model.eval()
        p_model.model.eval()
        R_1_dict, R_2_dict, y_dict, pre_dict, right_pre, wrong_pre = [], [], [], [], 0, 0
        with torch.no_grad():
            for x, y_solver, y in self.data_dict['test_loader']:
                x, y_solver, y = x.cuda(), y_solver.cuda(), y.cuda()
                out_d = dd_model.model(x).view(y.shape[0], self.S, self.S, self.T)
                out_p = p_model.model(x, y_solver).view(y.shape[0], self.S, self.S, self.T) + y_solver.view(y.shape[0], self.S, self.S, self.T)
                out_UQ_dp = self.evaluate_uncertainty(x, out_d, out_p)            
                if (self.model_s == 'FNO_sample_RE') or (self.model_s == 'sample_RE') or (self.model_s == 'pixel_NLL'):
                    R_1, R_2 = self.decode_all_RE(out_d, out_p, y)
                else:
                    R_1, R_2 = self.decode_all(out_d, out_p, y)
                    for _ in range(3):
                        R_1 = torch.mean(R_1, dim=-1)
                        R_2 = torch.mean(R_2, dim=-1)  
                for UQ_acc_index in range(R_1.shape[0]):
                    R_1_dict.append(R_1[UQ_acc_index].item())
                    R_2_dict.append(R_2[UQ_acc_index].item())
                    y_dict.append((R_1 - R_2)[UQ_acc_index].item())
                    pre_dict.append(out_UQ_dp[UQ_acc_index].item())
                    if ((R_1 - R_2)[UQ_acc_index]*out_UQ_dp[UQ_acc_index]) >= 0:
                        right_pre += 1
                    else:
                        wrong_pre += 1
        R_1_dict, R_2_dict, y_dict, pre_dict = np.array(R_1_dict), np.array(R_2_dict), np.array(y_dict), np.array(pre_dict)
        sort_index = np.argsort(y_dict)
        R_1_dict, R_2_dict, y_dict, pre_dict = R_1_dict[sort_index], R_2_dict[sort_index], y_dict[sort_index], pre_dict[sort_index]
        acc = right_pre / (right_pre + wrong_pre)
        return R_1_dict, R_2_dict, y_dict, pre_dict, acc
    
    def UQ_acc_plot(self, dd_model, p_model, epoch=1):
        R_1_dict, R_2_dict, y_dict, pre_dict, acc = self.UQ_acc_test(dd_model, p_model)
        UQ_acc_plot_x = np.linspace(1, R_1_dict.shape[0], R_1_dict.shape[0])
        sizes = np.random.randint(5, 20, size=R_1_dict.shape[0])
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.scatter(UQ_acc_plot_x, y_dict, label=f"r1-r2 truth", s=sizes, alpha=0.5)
        plt.scatter(UQ_acc_plot_x, pre_dict, label=f"r1-r2 pre acc:{acc:.6f}", s=sizes, alpha=0.5)
        plt.xlabel("Data Index")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.scatter(UQ_acc_plot_x, R_1_dict, label='r1', s=sizes, alpha=0.5)
        plt.scatter(UQ_acc_plot_x, R_2_dict, label='r2', s=sizes, alpha=0.5)
        plt.legend()
        plt.xlabel("Data Index")
        plt.ylabel("Loss")
        plt.subplot(1, 3, 3)
        # 对A进行排序并获取排序后的索引  
        sorted_indices = np.argsort(y_dict)  
        A_sorted = y_dict[sorted_indices]  
        B_sorted = pre_dict[sorted_indices]  
        # 将A分为20份并计算每份的组内均值  
        num_bins = 20  
        bin_size = y_dict.shape[0] // num_bins  
        A_means = np.array([A_sorted[i:i + bin_size].mean() for i in range(0, y_dict.shape[0], bin_size)])  
        # 计算每份中向量A和向量B对应位置的同号比率  
        same_sign_ratios = np.array([(np.sign(A_sorted[i:i + bin_size]) == np.sign(B_sorted[i:i + bin_size])).mean()   
                                    for i in range(0, y_dict.shape[0], bin_size)])
        plt.bar(A_means, same_sign_ratios, width=0.1, label='Accuracy')
        plt.legend()
        plt.xlabel("Value of Residual1-Residual2")
        plt.ylabel("Intra-group Accuracy")
        plt.savefig(f"{self.modelConfig['MODEL_PATH']}/figure/{epoch}_UQ.png")
        plt.show()
        plt.close()
        return acc
    
    def evaluate_uncertainty(self, x, out_d, out_p):
        # x: (B, H, W, T_in), out_d/out_p: (B, H, W, T) 
        # input of UQ model (after normalization)
        x_adv_UQ_dp = torch.concat([x, out_d, out_p], dim=3).cuda()
        # output of UQ model and calculate confidence
        self.model.eval()
        if self.model_s == 'FNO':
            R_1_pre, R_2_pre = self.model(x_adv_UQ_dp)
            R_1_pre, R_2_pre = self.data_dict['y_normalizer_cuda'].decode(R_1_pre), self.data_dict['y_normalizer_cuda'].decode(R_2_pre)
            for _ in range(3):
                R_1_pre = torch.mean(R_1_pre, dim=-1)
                R_2_pre = torch.mean(R_2_pre, dim=-1)
        out_UQ_dp = torch.zeros_like(R_1_pre)
        Judge_R_1 = R_1_pre - self.modelConfig['dual_judge']*R_2_pre
        Judge_R_2 = self.modelConfig['dual_judge']*R_1_pre - R_2_pre
        out_UQ_dp[Judge_R_1 > 0] = Judge_R_1[Judge_R_1 > 0]
        out_UQ_dp[Judge_R_2 < 0] = Judge_R_2[Judge_R_2 < 0]
        return out_UQ_dp


