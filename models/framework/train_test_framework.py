"""
@author: Xuerui Su
This file is the submain file to assist training and test
""" 
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
import torch
from data.seqrecord.module import MultiSourceDataModule, MultiSourceDataModule_UQ
from modules.climnet import ClimNet, ClimNet_UQ
from configs.Climax_train_modelparam import * # hyperparameters
from utils.metrics import lat_weighted_mse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import glob 

# torch.manual_seed(0)
# np.random.seed(0)

class UQ_Dual_Weather_Climai():
    def __init__(self) -> None:
        # model:
        self.Main_Model_1 = ClimNet(
                            const_vars=tuple(const_vars),
                            single_vars=tuple(single_vars),
                            atmos_vars=tuple(atmos_vars),
                            atmos_levels=tuple(atmos_levels),
                            patch_size=patch_size,
                            embed_dim=embed_dim,
                            depth=depth,
                            decoder_depth=decoder_depth,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            drop_path=drop_path,
                            drop_rate=drop_rate,
                            use_flash_attn=use_flash_attn,
                            )
        self.Main_Model_2 = ClimNet(
                            const_vars=tuple(const_vars),
                            single_vars=tuple(single_vars),
                            atmos_vars=tuple(atmos_vars),
                            atmos_levels=tuple(atmos_levels),
                            patch_size=patch_size,
                            embed_dim=embed_dim,
                            depth=depth,
                            decoder_depth=decoder_depth,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            drop_path=drop_path,
                            drop_rate=drop_rate,
                            use_flash_attn=use_flash_attn,
                            )
        self.UQ_Model = ClimNet_UQ(
                        const_vars=tuple(const_vars),
                        single_vars=tuple(single_vars),
                        atmos_vars=tuple(atmos_vars),
                        atmos_levels=tuple(atmos_levels),
                        patch_size=patch_size,
                        embed_dim=embed_dim,
                        depth=depth,
                        decoder_depth=decoder_depth,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path,
                        drop_rate=drop_rate,
                        use_flash_attn=use_flash_attn,
                        
                        embed_dim_UQ=embed_dim_UQ,
                        out_embed_dim_UQ=out_embed_dim_UQ,
                        depth_UQ=depth_UQ,
                        en_UQ_embed_dim=en_UQ_embed_dim,
                        num_heads_UQ_en=num_heads_UQ_en,
                        num_heads_UQ_down=num_heads_UQ_down,
                        mlp_ratio_UQ=mlp_ratio_UQ,
                        drop_path_UQ=drop_path_UQ,
                        drop_rate_UQ=drop_rate_UQ,
                        use_flash_attn_UQ=use_flash_attn_UQ,
                        decoder_depth_down_UQ=decoder_depth_down_UQ,
                        )
        # data:
        self.datamodule = MultiSourceDataModule(dict_root_dirs, dict_data_spatial_shapes, 
                                             dict_single_vars, dict_atmos_vars, dict_hrs_each_step, 
                                             dict_max_predict_range, batch_size, dict_metadata_dirs, 
                                             shuffle_buffer_size=shuffle_buffer_size, 
                                             val_shuffle_buffer_size=val_shuffle_buffer_size, 
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             use_old_loader=use_old_loader)
        self.datamodule.setup()
        self.train_dataloader_Original = self.datamodule.train_dataloader()
        self.test_dataloader_Original = self.datamodule.test_dataloader()
        self.val_dataloader_Original = self.datamodule.val_dataloader()
        self.datamodule_UQ = MultiSourceDataModule_UQ(dict_root_dirs, dict_data_spatial_shapes, 
                                             dict_single_vars, dict_atmos_vars, dict_hrs_each_step, 
                                             dict_max_predict_range, batch_size, dict_metadata_dirs, 
                                             shuffle_buffer_size=shuffle_buffer_size, 
                                             val_shuffle_buffer_size=val_shuffle_buffer_size, 
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             use_old_loader=use_old_loader)
        self.datamodule_UQ.setup()
        self.train_dataloader_UQ = self.datamodule_UQ.train_dataloader()
        self.test_dataloader_UQ = self.datamodule_UQ.test_dataloader()
        self.val_dataloader_UQ = self.datamodule_UQ.val_dataloader()
        # Trainer:
        lr_monitor = LearningRateMonitor(logging_interval=logging_interval)  # 或者 'epoch'，根据需要进行设置  
        checkpoint_callback = ModelCheckpoint(  
            dirpath=dirpath,  # saving checkpoint dir  
            filename=filename,  # checkpoints file name
            save_top_k=save_top_k,  # save top k models
            verbose=verbose,  # Only print news when save models
            monitor=monitor_param,  # use which loss to judge model 
            mode=mode,  # val_loss min is better  
            save_last=mode,  # save the last model too
            auto_insert_metric_name=auto_insert_metric_name,
        )  
        richmodelsummary = RichModelSummary(max_depth=max_depth)
        logger = TensorBoardLogger(save_dir=default_checkpoints_dir, version=1, name="lightning_logs")
        self.trainer_UQ = Trainer(accelerator=accelerator, devices=devices, max_epochs=max_epochs, 
                        enable_checkpointing=enable_checkpointing, strategy=strategy, 
                        logger=logger, precision=precision, num_nodes=num_nodes, 
                        callbacks=[checkpoint_callback, lr_monitor, richmodelsummary]) 
        # self.optimizer_UQ = torch.optim.AdamW(self.UQ_Model.parameters(), lr=lr_UQ, 
        #                                       weight_decay=weight_decay_UQ, betas=(beta_1_UQ, beta_2_UQ))
        # self.Schedule_UQ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_UQ, mode='min')
        os.makedirs(default_root_dir, exist_ok=True)
        prev_ckpts = glob.glob(os.path.join(default_root_dir, "checkpoints_UQ", "*.ckpt"))
        if len(prev_ckpts) > 0:
            self.resume_from_checkpoint_UQ = os.path.join(
                default_root_dir, "checkpoints_UQ", "last.ckpt"
            )
        else:
            self.resume_from_checkpoint_UQ = None
        self.Main_Model_Load(Main_Model_Path_1, Main_Model_Path_2)
        self.UQ_Model_Load(UQ_Model_path)

    def data_iterative_generation_Functest(self, ):
        for single_inputs, atmos_inputs, single_outputs, atmos_outputs, lead_times, metadata in self.train_dataloader_Original:
            _, single_pre_1, atmos_pre_1 = self.Main_Model_1(single_inputs, atmos_inputs, lead_times, metadata)
            _, single_pre_2, atmos_pre_2 = self.Main_Model_2(single_inputs, atmos_inputs, lead_times, metadata)
            break
        return [single_inputs, atmos_inputs, single_outputs, atmos_outputs, lead_times, metadata, 
                single_pre_1, atmos_pre_1, single_pre_2, atmos_pre_2]
        
    def UQ_data_iterative_generation_Functest(self, ):
        for single_inputs, atmos_inputs, single_outputs, atmos_outputs, lead_times, metadata, single_pre_1, atmos_pre_1, single_pre_2, atmos_pre_2 in self.train_dataloader_UQ:
            _, single_pre_1_o, atmos_pre_1_o = self.Main_Model_1(single_inputs, atmos_inputs, lead_times, metadata)
            _, single_pre_2_o, atmos_pre_2_o = self.Main_Model_2(single_inputs, atmos_inputs, lead_times, metadata)
            break
        return [single_inputs, atmos_inputs, single_outputs, atmos_outputs, lead_times, metadata, 
                single_pre_1, atmos_pre_1, single_pre_2, atmos_pre_2, 
                single_pre_1_o, atmos_pre_1_o, single_pre_2_o, atmos_pre_2_o]

    def UQ_Model_Training(self, ):
        self.trainer_UQ.fit(self.UQ_Model, datamodule=self.datamodule_UQ, ckpt_path=self.resume_from_checkpoint_UQ,) 
        
    def Dual_Main_Model_Trainer(self, ):
        pass
        
    def Main_Model_Load(self, path_1, path_2):
        self.Main_Model_1 = torch.load(path_1)
        self.Main_Model_2 = torch.load(path_2)
        
    def UQ_Model_Load(self, path):
        self.UQ_Model = torch.load(path)
        
    






# class dual_enhance_sample_compare_UQ():
#     def __init__(self, modelConfig):
#         # dual enhance parameters
#         self.modelConfig = modelConfig
#         self.save_main_every_epoch = 30
#         self.epoch = 0
#         self.datadriven_model = datadriven_model_weather("model_weights_init.pth")
#         self.physics_model = datadriven_model_weather("model_weights_init_2.pth")
#         self.dual_enhance_class = dual_enhance(modelConfig)
#         self.start_dual_epoch = modelConfig['start_dual_epoch']
#         self.writer = SummaryWriter(modelConfig['tensorboard_path'])

#     def save_dd_model_func(self):
#         torch.save(self.datadriven_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_dd")
#         print('dadadriven model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
#         gc.collect()

#     def save_p_model_func(self):
#         torch.save(self.physics_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_p")
#         print('physics model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
#         gc.collect()
        
#     def save_UQ_func(self):
#         # use this func carefully cause the occupation of buffer is very very big.
#         with open(f"{self.modelConfig['MODEL_PATH']}/epoch-{self.epoch}_FNO_buffer.pkl", 'wb') as buffer_f:
#             pickle.dump(self.dual_enhance_class.UQ_model.buffer, buffer_f)
#         print('buffer saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
#         torch.save(self.dual_enhance_class.UQ_model.model, f"{self.modelConfig['MODEL_PATH']}/ep_{self.epoch}_FNO_UQ")
#         print('UQ model saved at', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
#         gc.collect()
                
#     def datadriven_model_train(self):
#         self.datadriven_model.fit()
#         self.save_dd_model_func()
    
#     def physics_model_train(self):
#         self.physics_model.fit()
#         self.save_p_model_func()
    
#     def main_model_train(self):
#         self.datadriven_model_train()
#         self.physics_model_train()
    
#     def UQ_model_train(self, EPOCH=1):
        
#         self.save_UQ_func()
            
#     def dual_enhance(self, ):
#         if self.epoch >= self.start_dual_epoch:
#             self.main_model_evaluate(dual_before='dual_before') 
#             t1 = default_timer()
#             train_l2_dual_dd, train_l2_dual_p, opt_dd_num, opt_p_num = self.dual_enhance_class.dual_enhance(self.datadriven_model, self.physics_model)
#             t2 = default_timer()
#             self.writer.add_scalar('dual_enhance/PTD_l2_loss', train_l2_dual_dd, self.epoch)
#             self.writer.add_scalar('dual_enhance/DTP_l2_loss', train_l2_dual_p, self.epoch)
#             self.writer.add_scalar('dual_enhance/PTD_num', opt_dd_num, self.epoch)
#             self.writer.add_scalar('dual_enhance/DTP_num', opt_p_num, self.epoch)
#             with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
#                 print("Dual EPOCH:", self.epoch, "Time:", t2-t1, "Train No.ii Loss dd:", train_l2_dual_dd, 
#                     "Train No.ii Loss p:", train_l2_dual_p, "PTD", opt_dd_num, "DTP", opt_p_num, file=f)
#                 print("Dual EPOCH:", self.epoch, "Time:", t2-t1, "Train No.ii Loss dd:", train_l2_dual_dd, 
#                     "Train No.ii Loss p:", train_l2_dual_p, "PTD", opt_dd_num, "DTP", opt_p_num)    
#             self.main_model_evaluate(dual_before='dual_after') 
            
#     def main_model_evaluate(self, dual_before='dual_before'):
#         pass
    
#     def UQ_acc_plot(self, epoch):
#         if self.epoch >= self.start_dual_epoch:
#             acc = self.dual_enhance_class.UQ_model.UQ_acc_plot(self.datadriven_model, self.physics_model, epoch)
#             self.writer.add_scalar(f"UQ_acc/accuracy", acc, self.epoch)
    
#     def multi_pre_plot(self, step=10, epoch=1):
#         test_mse_step_dd, test_l2_step_dd = self.datadriven_model.multi_pre_plot(step=step, epoch=epoch)
#         with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
#             print("multi pre error dd EPOCH:", self.epoch, "MSE Error", test_mse_step_dd, "L2 Error", test_l2_step_dd)
#             print("multi pre error dd EPOCH:", self.epoch, "MSE Error", test_mse_step_dd, "L2 Error", test_l2_step_dd, file=f)
#         test_mse_step_p, test_l2_step_p = self.physics_model.multi_pre_plot(step=step, epoch=epoch)
#         with open(f"{self.modelConfig['MODEL_PATH']}/log.txt", "a+") as f:
#             print("multi pre error p EPOCH:", self.epoch, "MSE Error", test_mse_step_p, "L2 Error", test_l2_step_p)
#             print("multi pre error p EPOCH:", self.epoch, "MSE Error", test_mse_step_p, "L2 Error", test_l2_step_p, file=f)
#         self.writer.add_scalar(f"multi_pre/dd_mse_loss_step1", test_mse_step_dd[0], self.epoch)
#         self.writer.add_scalar(f"multi_pre/dd_l2_loss_step1", test_l2_step_dd[0], self.epoch)
#         self.writer.add_scalar(f"multi_pre/p_mse_loss_step1", test_mse_step_p[0], self.epoch)
#         self.writer.add_scalar(f"multi_pre/p_l2_loss_step1", test_l2_step_p[0], self.epoch)
        
#         self.writer.add_scalar(f"multi_pre/dd_mse_loss_step{int(step/2)}", test_mse_step_dd[int(step/2)], self.epoch)
#         self.writer.add_scalar(f"multi_pre/dd_l2_loss_step{int(step/2)}", test_l2_step_dd[int(step/2)], self.epoch)
#         self.writer.add_scalar(f"multi_pre/p_mse_loss_step{int(step/2)}", test_mse_step_p[int(step/2)], self.epoch)
#         self.writer.add_scalar(f"multi_pre/p_l2_loss_step{int(step/2)}", test_l2_step_p[int(step/2)], self.epoch)
        
#         self.writer.add_scalar(f"multi_pre/dd_mse_loss_step{step}", test_mse_step_dd[step-1], self.epoch)
#         self.writer.add_scalar(f"multi_pre/dd_l2_loss_step{step}", test_l2_step_dd[step-1], self.epoch)
#         self.writer.add_scalar(f"multi_pre/p_mse_loss_step{step}", test_mse_step_p[step-1], self.epoch)
#         self.writer.add_scalar(f"multi_pre/p_l2_loss_step{step}", test_l2_step_p[step-1], self.epoch)
        
#         plt.figure(figsize=(10, 6))
#         plt.subplot(1,2,1)
#         plt.plot(test_mse_step_dd, label='MSE Error Data Driven')
#         plt.plot(test_mse_step_p, label='MSE Error Physics')
#         plt.legend()
#         plt.yscale('log')
#         plt.xlabel('Step')
#         plt.ylabel('MSE Error')
#         plt.subplot(1,2,2)
#         plt.plot(test_l2_step_dd, label='Relative Error Data Driven')
#         plt.plot(test_l2_step_p, label='Relative Error Physics')
#         plt.legend()
#         plt.yscale('log')
#         plt.xlabel('Step')
#         plt.ylabel('Relative Error')
#         plt.savefig(f"{self.modelConfig['MODEL_PATH']}/figure/{epoch}_multi_step.png")
        
        
        
        