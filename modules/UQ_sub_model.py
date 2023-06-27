import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.FNO_2d import FNO2d_UQ_mean, FNO2d_UQ, FNO2d_UQ_single_layer
# from FNO_2d import FNO2d_UQ_mean, FNO2d_UQ
torch.manual_seed(0)
np.random.seed(0)

  
class MultiHeadSelfAttention(nn.Module):  
    def __init__(self, H, W, num_heads=8):  
        super(MultiHeadSelfAttention, self).__init__()  
        self.num_heads = num_heads  
        self.H = H  
        self.W = W  
        self.head_dim = H // num_heads  
  
        assert self.head_dim * num_heads == H, "H must be divisible by num_heads"  
  
        self.qkv = nn.Linear(H, H * 3, bias=False)  
        self.fc_out = nn.Linear(H, H)  
  
    def forward(self, x):  
        B, _, _ = x.size()  
        qkv = self.qkv(x).view(B, -1, self.num_heads, 3 * self.head_dim).transpose(1, 2)  
        query, key, value = qkv.chunk(3, dim=-1)  
  
        attn_output = self.self_attention(query, key, value)  
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.H)  
  
        return self.fc_out(attn_output)  
  
    def self_attention(self, query, key, value):  
        energy = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        attention = torch.softmax(energy, dim=-1)  
        out = torch.matmul(attention, value)  
  
        return out  
    
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)
        
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        
        h = self.proj(h)
        
        return x + h
    
  

class AttentionModel(nn.Module):  
    def __init__(self, S, input_ch=3):  
        super(AttentionModel, self).__init__()  
        self.encoder = nn.Sequential(  
            nn.Conv2d(input_ch, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU()  
        )  
        self.attention = AttnBlock(64)
        self.decoder1 = nn.Sequential(  
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
        )   
        self.decoder2 = nn.Sequential(  
            nn.Linear(8*(S//8)*(S//8), 256),
            nn.ReLU(),  
            nn.Linear(256, 2),
        )  

    def forward(self, x):  
        x = self.encoder(x.permute(0, 3, 1, 2))
        attn_output = self.attention(x)  
        out = self.decoder1(attn_output) 
        out = self.decoder2(out.contiguous().view(out.size(0), -1))
        return out

class FNO2d_UQ_sample(nn.Module):
    def __init__(self, modes1, modes2, width, S, T_in=10, T=1):
        super(FNO2d_UQ_sample, self).__init__()
        self.model1 = FNO2d_UQ_mean(modes1, modes2, width, T_in=T_in, T=T)
        self.model2 = AttentionModel(S=S, input_ch=2)
        self.T = T
        self.T_in = T_in
        
    def forward(self, x):
        # x: (B,H,W,T_in+2*T)
        out_d, out_p = self.model1(x) # B,H,W,T
        x = self.model2(torch.concat([out_d, out_p], dim=-1))
        # x: (B, 2)
        return x[:, 0], x[:, 1]
    
class FNO2d_UQ_sample_mainmodel(nn.Module):
    def __init__(self, modes1, modes2, width, S, T_in=10):
        super(FNO2d_UQ_sample_mainmodel, self).__init__()
        self.model1 = FNO2d_UQ(modes1, modes2, width, T_in=T_in)
        self.model2 = AttentionModel(S=S, input_ch=3)
        self.T_in = T_in
        
    def forward(self, x, out_d, out_p):
        x = self.model1(x) # B,H,W,T
        x = self.model2(torch.concat([x, out_d, out_p], dim=-1))
        # x: (B, 2)
        return x[:, 0], x[:, 1]

class UQ_sample_mainmodel(nn.Module):
    def __init__(self, modes1, modes2, width, S, T_in=10):
        super(UQ_sample_mainmodel, self).__init__()
        # self.model1 = FNO2d_UQ(modes1, modes2, width, T_in=T_in)
        self.model2 = AttentionModel(S=S, input_ch=12)
        self.T_in = T_in
        
    def forward(self, x, out_d, out_p):
        # x = self.model1(x) # B,H,W,T
        x = self.model2(torch.concat([x, out_d, out_p], dim=-1))
        # x: (B, 2)
        return x[:, 0], x[:, 1]
    
class FNO2d_UQ_NLL(nn.Module):
    def __init__(self, modes1, modes2, width, T_in=10, T=1):
        super(FNO2d_UQ_NLL, self).__init__()
        self.model1 = FNO2d_UQ(modes1, modes2, width, T_in=T_in+T)
        self.model2 = FNO2d_UQ(modes1, modes2, width, T_in=T_in+T)
        self.max_logvar = 0.5
        self.min_logvar = -10

    def forward(self, x, out_d, out_p):
        # x: (B, H, W, T_in), out_d/out_p: (B, H, W, T)
        log_var_d = self.model1(torch.concat([x, out_d], dim=-1))
        log_var_d = self.max_logvar - F.softplus(self.max_logvar - log_var_d)
        log_var_d = self.min_logvar + F.softplus(log_var_d - self.min_logvar)
        
        log_var_p = self.model2(torch.concat([x, out_p], dim=-1))
        log_var_p = self.max_logvar - F.softplus(self.max_logvar - log_var_p)
        log_var_p = self.min_logvar + F.softplus(log_var_p - self.min_logvar)
        
        return log_var_d, log_var_p

class FNO2d_UQ_Ensemble(nn.Module):
    def __init__(self, ensemble_num=5, T_in=10, T=1):
        super(FNO2d_UQ_Ensemble, self).__init__()
        self.model_dict1, self.model_dict2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(ensemble_num):
            self.model_dict1.append(FNO2d_UQ_single_layer(4, 4, 10, T_in=T_in+T))
            self.model_dict2.append(FNO2d_UQ_single_layer(4, 4, 10, T_in=T_in+T))

    def cal_mean_var(self, data, mean, M2, n):
        n += 1
        delta = data - mean  
        mean += delta / n  
        delta2 = data - mean  
        M2 += delta * delta2 
        return mean, M2, n
    
    def forward(self, x, out_d, out_p):
        # x: (B, H, W, T_in), out_d/out_p: (B, H, W, T)
        mean_d, mean_p, var_d, var_p, UQ_Ensemble_index = 0.0, 0.0, 0.0, 0.0, 0
        for model_d in self.model_dict1:
            data_d = model_d(torch.concat([x, out_d], dim=-1))  
            mean_d, var_d, UQ_Ensemble_index = self.cal_mean_var(data_d, mean_d, var_d, UQ_Ensemble_index)
        for model_p in self.model_dict2:
            data_p = model_p(torch.concat([x, out_p], dim=-1))  
            mean_p, var_p, UQ_Ensemble_index = self.cal_mean_var(data_p, mean_p, var_p, UQ_Ensemble_index)
        if UQ_Ensemble_index < 2:  
            return float('nan')  
        else:  
            var_d = var_d / UQ_Ensemble_index
            var_p = var_p / UQ_Ensemble_index
            return mean_d, mean_p, var_d,  var_p

# # Example usage 
# # print the number of parameters
# from functools import reduce
# import operator
# def count_params(model):
#     c = 0
#     for p in list(model.parameters()):
#         c += reduce(operator.mul, 
#                     list(p.size()+(2,) if p.is_complex() else p.size()))
#     return c 
# B, H, W = 8, 64, 64  
# x = torch.randn(B, H, W, 10).cuda()
# out_d = torch.randn(B, H, W, 1).cuda()
# out_p = torch.randn(B, H, W, 1).cuda()
# model = FNO2d_UQ_sample_mainmodel(4, 4, 5, 64, 10).cuda()
# x1, x2 = model(x, out_d, out_p)
# print(x1.shape)  # Output: torch.Size([4, 2])  
# print("confidence quantification model parameters number:", count_params(model))
# 

