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
from models.main_model_class import data_load, datadriven_model, physics_model
import gc
import pickle
import datetime
from models.dual_enhance_class import dual_enhance
from torch.utils.tensorboard import SummaryWriter   
torch.manual_seed(0)
np.random.seed(0)

class dual_enhance_sample_compare_UQ():
    def __init__(self, modelConfig):
        # dual enhance parameters
        self.modelConfig = modelConfig
        self.save_main_every_epoch = 30
        self.epoch = 0
        self.datadriven_model = datadriven_model(modelConfig)
        self.physics_model = physics_model(modelConfig)
        self.dual_enhance_class = dual_enhance(modelConfig)
        self.start_dual_epoch = modelConfig['start_dual_epoch']
        self.writer = SummaryWriter(modelConfig['tensorboard_path'])
        
    def save_dd_model_func(self):
        if self.epoch % self.save_main_every_epoch == 0:
            torch.save(self.datadriven_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_dd")
            print('dadadriven model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            gc.collect()
            
    def save_p_model_func(self):
        if self.epoch % self.save_main_every_epoch == 0:
            torch.save(self.physics_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_p")
            print('physics model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            gc.collect()
        
    def save_UQ_func(self):
        if self.epoch % 50 == 0 and self.epoch >= self.start_dual_epoch:
            # use this func carefully cause the occupation of buffer is very very big.
            with open(f"{self.modelConfig['MODEL_PATH']}/epoch-{self.epoch}_FNO_buffer.pkl", 'wb') as buffer_f:
                pickle.dump(self.dual_enhance_class.UQ_model.buffer, buffer_f)
            print('buffer saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            torch.save(self.dual_enhance_class.UQ_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_UQ")
            print('UQ model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            gc.collect()
                
    def datadriven_model_train(self, epoch=1):
        for _ in range(epoch):
            t1 = default_timer()
            train_mse, train_l2 = self.datadriven_model.train_epoch()
            test_mse, test_l2 = self.datadriven_model.test()
            t2 = default_timer()
            self.writer.add_scalar('datadriven/dd_train_mse_loss', train_mse, self.epoch)
            self.writer.add_scalar('datadriven/dd_train_l2_loss', train_l2, self.epoch)
            self.writer.add_scalar('datadriven/dd_test_mse_loss', test_mse, self.epoch)
            self.writer.add_scalar('datadriven/dd_test_l2_loss', test_l2, self.epoch)
            with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
                print("Data Driven EPOCH:", self.epoch, "Time:", t2-t1, "Train No.i Loss:", train_mse, "Train No.ii Loss:", train_l2, 
                        "Test No.i Loss:", test_mse, "Test No.ii Loss:", test_l2)
                print("Data Driven EPOCH:", self.epoch, "Time:", t2-t1, "Train No.i Loss:", train_mse, "Train No.ii Loss:", train_l2, 
                        "Test No.i Loss:", test_mse, "Test No.ii Loss:", test_l2, file=f)
        self.save_dd_model_func()
    
    def physics_model_train(self, epoch=1):
        for _ in range(epoch):
            t1 = default_timer()
            train_mse, train_l2, train_mse_solver, train_l2_solver = self.physics_model.train_epoch()
            test_mse, test_l2, test_mse_solver, test_l2_solver = self.physics_model.test()
            t2 = default_timer()
            self.writer.add_scalar('physics/p_train_mse_loss', train_mse, self.epoch)
            self.writer.add_scalar('physics/p_train_l2_loss', train_l2, self.epoch)
            self.writer.add_scalar('physics/p_train_mse_solver_loss', train_mse_solver, self.epoch)
            self.writer.add_scalar('physics/p_train_l2_solver_loss', train_l2_solver, self.epoch)
            self.writer.add_scalar('physics/p_test_mse_loss', test_mse, self.epoch)
            self.writer.add_scalar('physics/p_test_l2_loss', test_l2, self.epoch)
            self.writer.add_scalar('physics/p_test_mse_solver_loss', test_mse_solver, self.epoch)
            self.writer.add_scalar('physics/p_test_l2_solver_loss', test_l2_solver, self.epoch)
            with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
                print("Physics Residual EPOCH:", self.epoch, "Time:", t2-t1, 
                        "Train No.i Loss:", train_mse, "Train No.ii Loss:", train_l2, 
                        "Train No.i Solver Loss:", train_mse_solver, "Train No.ii Solver Loss:", train_l2_solver, 
                        "Test No.i Loss:", test_mse, "Test No.ii Loss:", test_l2, 
                        "Test No.i Solver Loss:", test_mse_solver, "Test No.ii Solver Loss:", test_l2_solver)
                print("Physics Residual EPOCH:", self.epoch, "Time:", t2-t1, 
                        "Train No.i Loss:", train_mse, "Train No.ii Loss:", train_l2, 
                        "Train No.i Solver Loss:", train_mse_solver, "Train No.ii Solver Loss:", train_l2_solver, 
                        "Test No.i Loss:", test_mse, "Test No.ii Loss:", test_l2, 
                        "Test No.i Solver Loss:", test_mse_solver, "Test No.ii Solver Loss:", test_l2_solver, file=f)
        self.save_p_model_func()
    
    def main_model_train(self, epoch=1):
        for _ in range(epoch):
            self.datadriven_model_train()
            self.physics_model_train()
    
    def UQ_model_train(self, EPOCH=1):
        if self.epoch >= self.start_dual_epoch:
            for _ in range(EPOCH):
                t1 = default_timer()
                train_mse, train_l2, train_rank_loss = self.dual_enhance_class.UQ_model.train_epoch(self.datadriven_model, self.physics_model)
                test_mse, test_l2, test_rank_loss = self.dual_enhance_class.UQ_model.test(self.datadriven_model, self.physics_model)
                t2 = default_timer()  
                self.writer.add_scalar('UQ/UQ_train_mse_loss', train_mse, self.epoch)
                self.writer.add_scalar('UQ/UQ_train_l2_loss', train_l2, self.epoch)
                self.writer.add_scalar('UQ/UQ_train_rank_loss', train_rank_loss, self.epoch)
                self.writer.add_scalar('UQ/UQ_test_mse_loss', test_mse, self.epoch)
                self.writer.add_scalar('UQ/UQ_test_l2_loss', test_l2, self.epoch)
                self.writer.add_scalar('UQ/UQ_test_rank_loss', test_rank_loss, self.epoch)
                with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
                    print("UQ EPOCH:", self.epoch, "Time:", t2-t1, 
                        "Train No.i Loss:", train_mse, "Train No.ii Loss:", train_l2, "Train Rank Loss:", train_rank_loss, 
                        "Test No.i Loss:", test_mse, "Test No.ii Loss: ", test_l2, "Test Rank Loss:", test_rank_loss, file=f)
                    print("UQ EPOCH:", self.epoch, "Time:", t2-t1, 
                        "Train No.i Loss:", train_mse, "Train No.ii Loss:", train_l2, "Train Rank Loss:", train_rank_loss, 
                        "Test No.i Loss:", test_mse, "Test No.ii Loss: ", test_l2, "Test Rank Loss:", test_rank_loss)
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
        t1 = default_timer()
        test_mse, test_l2 = self.datadriven_model.test()
        t2 = default_timer()
        self.writer.add_scalar(f"dual_evaluate/dd_mse_loss_{dual_before}", test_mse, self.epoch)
        self.writer.add_scalar(f"dual_evaluate/dd_l2_loss_{dual_before}", test_l2, self.epoch)
        with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
            print("Evaluate Data Driven EPOCH:", self.epoch, "Time:", t2-t1, "Test No.i Loss:", test_mse, "Test No.ii Loss:", test_l2)
            print("Evaluate Data Driven EPOCH:", self.epoch, "Time:", t2-t1, "Test No.i Loss:", test_mse, "Test No.ii Loss:", test_l2, file=f)
        t1 = default_timer()
        test_mse, test_l2, test_mse_solver, test_l2_solver = self.physics_model.test()
        t2 = default_timer()
        self.writer.add_scalar(f"dual_evaluate/p_mse_loss_{dual_before}", test_mse, self.epoch)
        self.writer.add_scalar(f"dual_evaluate/p_l2_loss_{dual_before}", test_l2, self.epoch)
        self.writer.add_scalar(f"dual_evaluate/p_mse_solver_loss_{dual_before}", test_mse_solver, self.epoch)
        self.writer.add_scalar(f"dual_evaluate/p_l2_solver_loss_{dual_before}", test_l2_solver, self.epoch)
        with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
            print("Evaluate Physics Residual EPOCH:", self.epoch, "Time:", t2-t1, "Test No.i Loss:", test_mse, "Test No.ii Loss:", test_l2, 
                  "Test No.i Solver Loss:", test_mse_solver, "Test No.ii Solver Loss:", test_l2_solver)
            print("Evaluate Physics Residual EPOCH:", self.epoch, "Time:", t2-t1, "Test No.i Loss:", test_mse, "Test No.ii Loss:", test_l2, 
                  "Test No.i Solver Loss:", test_mse_solver, "Test No.ii Solver Loss:", test_l2_solver, file=f)

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
        
        
        
        