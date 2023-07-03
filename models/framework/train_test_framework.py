"""
@author: Xuerui Su
This file is the submain file to assist training and test
""" 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from models.framework.main_model_class import data_load, datadriven_model, physics_model
from models.framework.Climai_train import datadriven_model_weather
import gc
import pickle
import datetime
from models.DualEnhance.dual_enhance_class import dual_enhance
from torch.utils.tensorboard import SummaryWriter   
# torch.manual_seed(0)
# np.random.seed(0)

class dual_enhance_sample_compare_UQ():
    def __init__(self, modelConfig):
        # dual enhance parameters
        self.modelConfig = modelConfig
        self.save_main_every_epoch = 30
        self.epoch = 0
        self.datadriven_model = datadriven_model_weather("model_weights_init.pth")
        self.physics_model = datadriven_model_weather("model_weights_init_2.pth")
        self.dual_enhance_class = dual_enhance(modelConfig)
        self.start_dual_epoch = modelConfig['start_dual_epoch']
        self.writer = SummaryWriter(modelConfig['tensorboard_path'])

    def save_dd_model_func(self):
        torch.save(self.datadriven_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_dd")
        print('dadadriven model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        gc.collect()

    def save_p_model_func(self):
        torch.save(self.physics_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_p")
        print('physics model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        gc.collect()
        
    def save_UQ_func(self):
        # use this func carefully cause the occupation of buffer is very very big.
        with open(f"{self.modelConfig['MODEL_PATH']}/epoch-{self.epoch}_FNO_buffer.pkl", 'wb') as buffer_f:
            pickle.dump(self.dual_enhance_class.UQ_model.buffer, buffer_f)
        print('buffer saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        torch.save(self.dual_enhance_class.UQ_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_UQ")
        print('UQ model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        gc.collect()
                
    def datadriven_model_train(self):
        self.datadriven_model.fit()
        self.save_dd_model_func()
    
    def physics_model_train(self):
        self.physics_model.fit()
        self.save_p_model_func()
    
    def main_model_train(self):
        self.datadriven_model_train()
        self.physics_model_train()
    
    def UQ_model_train(self, EPOCH=1):
        
        self.save_UQ_func()
            
    def dual_enhance(self, ):
        if self.epoch >= self.start_dual_epoch:
            self.main_model_evaluate(dual_before='dual_before') 
            t1 = default_timer()
            train_l2_dual_dd, train_l2_dual_p, opt_dd_num, opt_p_num = self.dual_enhance_class.dual_enhance(self.datadriven_model, self.physics_model)
            t2 = default_timer()
            self.writer.add_scalar('dual_enhance/PTD_l2_loss', train_l2_dual_dd, self.epoch)
            self.writer.add_scalar('dual_enhance/DTP_l2_loss', train_l2_dual_p, self.epoch)
            self.writer.add_scalar('dual_enhance/PTD_num', opt_dd_num, self.epoch)
            self.writer.add_scalar('dual_enhance/DTP_num', opt_p_num, self.epoch)
            with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
                print("Dual EPOCH:", self.epoch, "Time:", t2-t1, "Train No.ii Loss dd:", train_l2_dual_dd, 
                    "Train No.ii Loss p:", train_l2_dual_p, "PTD", opt_dd_num, "DTP", opt_p_num, file=f)
                print("Dual EPOCH:", self.epoch, "Time:", t2-t1, "Train No.ii Loss dd:", train_l2_dual_dd, 
                    "Train No.ii Loss p:", train_l2_dual_p, "PTD", opt_dd_num, "DTP", opt_p_num)    
            self.main_model_evaluate(dual_before='dual_after') 
            
    def main_model_evaluate(self, dual_before='dual_before'):
        pass
    
    def UQ_acc_plot(self, epoch):
        if self.epoch >= self.start_dual_epoch:
            acc = self.dual_enhance_class.UQ_model.UQ_acc_plot(self.datadriven_model, self.physics_model, epoch)
            self.writer.add_scalar(f"UQ_acc/accuracy", acc, self.epoch)
    
    def multi_pre_plot(self, step=10, epoch=1):
        test_mse_step_dd, test_l2_step_dd = self.datadriven_model.multi_pre_plot(step=step, epoch=epoch)
        with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
            print("multi pre error dd EPOCH:", self.epoch, "MSE Error", test_mse_step_dd, "L2 Error", test_l2_step_dd)
            print("multi pre error dd EPOCH:", self.epoch, "MSE Error", test_mse_step_dd, "L2 Error", test_l2_step_dd, file=f)
        test_mse_step_p, test_l2_step_p = self.physics_model.multi_pre_plot(step=step, epoch=epoch)
        with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
            print("multi pre error p EPOCH:", self.epoch, "MSE Error", test_mse_step_p, "L2 Error", test_l2_step_p)
            print("multi pre error p EPOCH:", self.epoch, "MSE Error", test_mse_step_p, "L2 Error", test_l2_step_p, file=f)
        self.writer.add_scalar(f"multi_pre/dd_mse_loss_step1", test_mse_step_dd[0], self.epoch)
        self.writer.add_scalar(f"multi_pre/dd_l2_loss_step1", test_l2_step_dd[0], self.epoch)
        self.writer.add_scalar(f"multi_pre/p_mse_loss_step1", test_mse_step_p[0], self.epoch)
        self.writer.add_scalar(f"multi_pre/p_l2_loss_step1", test_l2_step_p[0], self.epoch)
        
        self.writer.add_scalar(f"multi_pre/dd_mse_loss_step{int(step/2)}", test_mse_step_dd[int(step/2)], self.epoch)
        self.writer.add_scalar(f"multi_pre/dd_l2_loss_step{int(step/2)}", test_l2_step_dd[int(step/2)], self.epoch)
        self.writer.add_scalar(f"multi_pre/p_mse_loss_step{int(step/2)}", test_mse_step_p[int(step/2)], self.epoch)
        self.writer.add_scalar(f"multi_pre/p_l2_loss_step{int(step/2)}", test_l2_step_p[int(step/2)], self.epoch)
        
        self.writer.add_scalar(f"multi_pre/dd_mse_loss_step{step}", test_mse_step_dd[step-1], self.epoch)
        self.writer.add_scalar(f"multi_pre/dd_l2_loss_step{step}", test_l2_step_dd[step-1], self.epoch)
        self.writer.add_scalar(f"multi_pre/p_mse_loss_step{step}", test_mse_step_p[step-1], self.epoch)
        self.writer.add_scalar(f"multi_pre/p_l2_loss_step{step}", test_l2_step_p[step-1], self.epoch)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1,2,1)
        plt.plot(test_mse_step_dd, label='MSE Error Data Driven')
        plt.plot(test_mse_step_p, label='MSE Error Physics')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('MSE Error')
        plt.subplot(1,2,2)
        plt.plot(test_l2_step_dd, label='Relative Error Data Driven')
        plt.plot(test_l2_step_p, label='Relative Error Physics')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('Relative Error')
        plt.savefig(f"{self.modelConfig['MODEL_PATH']}/figure/{epoch}_multi_step.png")
        
        
        
        