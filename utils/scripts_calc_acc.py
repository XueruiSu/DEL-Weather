"""
@author: Xuerui Su
calculate the acc
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def calc_acc(model, test_residual2, test_loader2, compute_accuracy):
    model.eval()
    pred = torch.zeros(test_residual2.shape)
    index = 0
    with torch.no_grad():
        for x, y in test_loader2:
            x, y = x.cuda(), y.cuda()
            
            out = model(x).view(y.shape[0], y.shape[1], y.shape[2], y.shape[3])
            out = torch.exp(out)
            
            pred[index*y.shape[0]:(index+1)*y.shape[0]] = out.detach().cpu()

            index = index + 1

    # calculate acc
    pred_square = pred**2
    test_residual_square = test_residual2**2

    test_residual_compare = torch.zeros(test_residual2.shape)
    test_residual_compare[:-1] = test_residual_square[1:] - test_residual_square[:-1]
    test_residual_compare[-1] = test_residual_square[0] - test_residual_square[-1]

    test_residual_01 = torch.ones_like(test_residual_compare)  # 创建一个与x形状相同的全1张量
    test_residual_01[test_residual_compare < 0] = 0  # 将小于等于0的元素赋值为0

    test_pred_compare = torch.zeros(test_residual2.shape)
    test_pred_compare[:-1] = pred_square[1:] - pred_square[:-1]
    test_pred_compare[-1] = pred_square[0] - pred_square[-1]

    pred_01 = torch.ones_like(test_pred_compare)  # 创建一个与x形状相同的全1张量
    pred_01[test_pred_compare < 0] = 0  # 将小于等于0的元素赋值为0


    # 计算预测准确率
    print(f"单个数据预测准确率：", compute_accuracy(test_residual_01[0], pred_01[0]))
    print(f"平均预测准确率：", compute_accuracy(test_residual_01, pred_01))
    
    return compute_accuracy(test_residual_01, pred_01)


def Ranking_loss(pre, res_square):
    # pre: B, H, W, T
    # res_square: B, H, W, T
    middle = int(pre.shape[0]/2)
    
    pre_1, pre_2 = pre[:middle], pre[middle:]
    for i in range(3):
        pre_1, pre_2 = torch.mean(pre_1, dim=1), torch.mean(pre_2, dim=1)
    compare_pre = (pre_1 - pre_2)

    res_square_1, res_square_2 = res_square[:middle], res_square[middle:]
    for i in range(3):
        res_square_1, res_square_2 = torch.mean(res_square_1, dim=1), torch.mean(res_square_2, dim=1)
    compara_truth = (res_square_1 - res_square_2)
    
    rank_loss1 = -torch.log(1 / (1 + torch.exp(-compare_pre[compara_truth >= 0])))*2*torch.exp(-torch.abs(compara_truth[compara_truth >= 0])*8).detach()
    rank_loss2 = -torch.log(1- 1 / (1 + torch.exp(-compare_pre[compara_truth < 0])))*2*torch.exp(-torch.abs(compara_truth[compara_truth < 0])*8).detach()
    # print(rank_loss1, rank_loss2)
    if rank_loss1.size()[0] == 0:
        rank_loss = (rank_loss2.sum())
    elif rank_loss2.size()[0] == 0:
        rank_loss = (rank_loss1.sum())
    else:
        rank_loss = (rank_loss1.sum() + rank_loss2.sum())
    return rank_loss / pre.shape[0]

def Ranking_loss2(R_1_pre, R_2_pre, R_1, R_2):
    # out_d/out_p: (B, H, W, T), R_1/R_2:(B, H, W, T)
    for _ in range(3):
        R_1, R_2 = torch.mean(R_1, dim=-1), torch.mean(R_2, dim=-1)
        R_1_pre, R_2_pre = torch.mean(R_1_pre, dim=-1), torch.mean(R_2_pre, dim=-1)
    pre = R_1_pre - R_2_pre
    truth = R_1 - R_2
    A = 1
    fre = 1
    rank_loss1 = -torch.log(1 / (1 + A*torch.exp(-fre*pre[truth >= 0])))
    rank_loss2 = -torch.log(1- 1 / (1 + A*torch.exp(-fre*pre[truth < 0])))
    if rank_loss1.size()[0] == 0:
        rank_loss = (rank_loss2.sum())
    elif rank_loss2.size()[0] == 0:
        rank_loss = (rank_loss1.sum())
    else:
        rank_loss = (rank_loss1.sum() + rank_loss2.sum())
    return rank_loss / R_1_pre.shape[0]

def Ranking_loss_subnet(R_1_pre, R_2_pre, R_1, R_2):
    # out: (B, 2), R_1/R_2: (B,)
    pre = R_1_pre - R_2_pre
    truth = R_1 - R_2
    A = 1
    fre = 1
    rank_loss1 = -torch.log(1 / (1 + A*torch.exp(-fre*pre[truth >= 0])))
    rank_loss2 = -torch.log(1- 1 / (1 + A*torch.exp(-fre*pre[truth < 0])))
    # print(rank_loss1, rank_loss2)
    if rank_loss1.size()[0] == 0:
        rank_loss = (rank_loss2.sum())
    elif rank_loss2.size()[0] == 0:
        rank_loss = (rank_loss1.sum())
    else:
        rank_loss = (rank_loss1.sum() + rank_loss2.sum())
    return rank_loss / R_1_pre.shape[0]
 

def compute_accuracy(truth, prediction, estimation, str_plot="0.png", MODEL_PATH='./'):
    # 将张量和预测张量拉平为一维张量
    truth_flat = truth
    prediction_flat = prediction
    estimation = (estimation - torch.mean(estimation)) / torch.std(estimation)
    
    # 对比两个张量的元素是否相同，得到四个布尔型张量
    true_positive = ((truth_flat == 1) & (prediction_flat == 1))
    true_negative = ((truth_flat == 0) & (prediction_flat == 0))
    false_positive = ((truth_flat == 1) & (prediction_flat == 0)) 
    false_negative = ((truth_flat == 0) & (prediction_flat == 1)) 

    # 计算预测准确率
    accuracy00 = true_positive.sum().item() / torch.ones_like(truth_flat).sum()
    accuracy11 = true_negative.sum().item() / torch.ones_like(truth_flat).sum()
    wrong_acc_01 = false_positive.sum().item() / torch.ones_like(truth_flat).sum()
    wrong_acc_10 = false_negative.sum().item() / torch.ones_like(truth_flat).sum()
    
    hist_num1 = estimation[true_positive].view(-1).numpy()
    hist_num2 = estimation[true_negative].view(-1).numpy()
    hist_num12 = np.concatenate([hist_num1, hist_num2])
    hist_num3 = estimation[false_positive].view(-1).numpy()
    hist_num4 = estimation[false_negative].view(-1).numpy()
    hist_num34 = np.concatenate([hist_num3, hist_num4])
    
    mean12 = np.mean((hist_num12))
    mean34 = np.mean((hist_num34))
    print("mean true", mean12, "mean false", mean34)
       
    plt.figure()
    plt.hist(hist_num12, density=True, bins=1000, histtype='step', label=f"True_{mean12}")
    plt.hist(hist_num34, density=True, bins=1000, histtype='step', label=f"False_{mean34}")
    plt.legend()
    plt.xlabel("Truth: Residual1-Residual2")
    plt.savefig(f"{MODEL_PATH}/figure/"+str_plot)
    plt.show()
    plt.close()
    
    return accuracy00, accuracy11, wrong_acc_01, wrong_acc_10

