"""
@author: 
UQ training with $u_{t+1}-\hat{u_{t+1}}$
"""


import torch
import numpy as np
import datetime
import os
import sys
from models.dual_enhance_func import UQ_test
import sys
sys.path.append("..")

torch.manual_seed(0)
np.random.seed(0)


################################################################
# configs
################################################################

import argparse  
# 带参数表示true，不带参数表示False.
def get_args(argv=None):  
    parser = argparse.ArgumentParser(description='Put your hyperparameters')  
    # (self.modelConfig['UQ_mse']*mse_loss + self.modelConfig['UQ_l2']*l2_loss + self.modelConfig['UQ_Rank']*rank_loss)
    
    parser.add_argument('-UQ_mse', '--UQ_mse', default=1.0, type=float, help='UQ mse loss coef')  
    parser.add_argument('-UQ_l2', '--UQ_l2', default=1.0, type=float, help='UQ l2 loss coef')  
    parser.add_argument('-UQ_Rank', '--UQ_Rank', default=1.0, type=float, help='UQ Ranking loss coef')  
    
    parser.add_argument('-UQ_repeat', '--UQ_repeat', default=400, type=int, help='UQ training repeatation')  
    parser.add_argument('-epsilon', '--epsilon', default=1e-5, type=float, help='poor data generation epsilon')  
    parser.add_argument('-Dual', '--Dual', action='store_true', help='dual or not')  
    parser.add_argument('-model_s', '--model_s', default='FNO', type=str, help='UQ model selection')  
    parser.add_argument('-physics_single_t', '--physics_single_t', action='store_true', help='physics_single_teach or not')  
    parser.add_argument('-adv_num', '--adv_num', default=1, type=int, help='advsarial generation number')  
    parser.add_argument('-visc', '--visc', default=0.001, type=float, help='viscosity number')  
    
    parser.add_argument('-V100', '--V100', action='store_true', help='run on 16G device or not')  
    parser.add_argument('-modes', '--modes', default=8, type=int, help='FNO hyper para')  
    parser.add_argument('-width', '--width', default=20, type=int, help='FNO hyper para')  
    parser.add_argument('-dual_judge', '--dual_judge', default=1, type=float, help='dual_judge')  
    
    parser.add_argument('-modes_UQ', '--modes_UQ', default=4, type=int, help='FNO hyper para modes')  
    parser.add_argument('-width_UQ', '--width_UQ', default=10, type=int, help='FNO hyper para width')  
    
    parser.add_argument('-start_dual_epoch', '--start_dual_epoch', default=300, type=int, help='start_dual_epoch')  
    parser.add_argument('-docu', '--docu', default='FNO_subnet', type=str, help='experiment document/tensorboard_dir')  
    parser.add_argument('-use_y', '--use_y', action='store_true', help='use_y or not')  
    
    parser.add_argument('-dual_opt_repeat', '--dual_opt_repeat', default=1, type=int, help='dual opt repeat times')  
    parser.add_argument('-dual_num', '--dual_num', default=20, type=int, help='dual num of batches')  
    
    parser.add_argument('-adv_generate', '--adv_generate', action='store_true', help='advserially generate new data or not')  
    
    parser.add_argument('-learning_rate_main', '--learning_rate_main', default=1e-4, type=float, help='learning_rate_main')  
    
    return parser.parse_args(argv)  
  
# args parser  
args = get_args(sys.argv[1:])  
print(args)  

modelConfig = {
        "BLOB_PATH": '/blob/xueruisu',
        "UQ_repeat": args.UQ_repeat,
        "UQ_mse": args.UQ_mse,
        "UQ_l2": args.UQ_l2,
        "UQ_Rank": args.UQ_Rank,
        "epsilon": args.epsilon,
        "Dual": args.Dual,
        "model_s": args.model_s,
        "physics_single_t": args.physics_single_t,
        "adv_num": args.adv_num,
        "visc": args.visc,
        "V100": args.V100,
        "modes": args.modes,
        "width": args.width,
        "dual_judge": args.dual_judge, 
        "modes_UQ": args.modes_UQ,
        "width_UQ": args.width_UQ, 
        "start_dual_epoch": args.start_dual_epoch, 
        "tensorboard_path": f"/blob/xueruisu/FNO_base_UQ_for_scale_model/DEL-Reform/tensorboard_log/{args.docu}/",
        "use_y": args.use_y,
        "dual_opt_repeat": args.dual_opt_repeat, 
        "dual_num": args.dual_num,
        "adv_generate": args.adv_generate,
        "path_d": "/blob/xueruisu/FNO_base_UQ_for_scale_model/DEL-Reform/log/code_reform_test/no_dual2023-06-05-04-37-52/ep_600_FNO_dd",
        "path_p": "/blob/xueruisu/FNO_base_UQ_for_scale_model/DEL-Reform/log/code_reform_test/no_dual2023-06-05-04-37-52/ep_600_FNO_p",
        }

# 获取当前日期和时间
now_time = datetime.datetime.now()
# 格式化输出日期和时间，精确到秒
time_str = now_time.strftime("%Y-%m-%d-%H-%M-%S")
# time_str = time_str[len('2023-'):len('2023-05-07')]

CODE_PATH = f"{modelConfig['BLOB_PATH']}/FNO_base_UQ_for_scale_model/DEL-Reform"
MODEL_PATH = CODE_PATH+'/log/code_reform_test/'+args.docu + time_str
modelConfig['MODEL_PATH'] = MODEL_PATH

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if not os.path.exists(MODEL_PATH+"/figure"):
    os.mkdir(MODEL_PATH+"/figure")

ntrain = 70
ntest = 30
batch_size = 8
epochs = 600
scheduler_step = 100
scheduler_gamma = 0.5
print(epochs, args.learning_rate_main, scheduler_step, scheduler_gamma)
modelConfig['ntrain'] = ntrain
modelConfig['ntest'] = ntest

modelConfig['batch_size_dd'] = batch_size
modelConfig['batch_size_p'] = batch_size
modelConfig['batch_size_UQ_train'] = 6
modelConfig['batch_size_dual'] = 4

modelConfig['epochs'] = epochs

modelConfig['learning_rate_dd'] = args.learning_rate_main
modelConfig['scheduler_step_dd'] = scheduler_step
modelConfig['scheduler_gamma_dd'] = scheduler_gamma

modelConfig['learning_rate_p'] = args.learning_rate_main
modelConfig['scheduler_step_p'] = scheduler_step
modelConfig['scheduler_gamma_p'] = scheduler_gamma

modelConfig['learning_rate_UQ'] = 1e-4
modelConfig['scheduler_step_UQ'] = scheduler_step
modelConfig['scheduler_gamma_UQ'] = scheduler_gamma

sub = 1
S = 64 // sub
modelConfig['S'] = S
# T_in+T <= 40
modelConfig['T_in'] = 10
modelConfig['T'] = 1


training_file_path = CODE_PATH + f"/res/solver_solution_train_v{modelConfig['visc']}.npz"
testing_file_path = CODE_PATH + f"/res/solver_solution_test_v{modelConfig['visc']}.npz"
modelConfig['training_file_path'] = training_file_path
modelConfig['testing_file_path'] = testing_file_path

# NSvisc_samplebase_compareUQ_train_test(modelConfig)

UQ_test(modelConfig)
# main_model_evaluate_function_test(modelConfig)



