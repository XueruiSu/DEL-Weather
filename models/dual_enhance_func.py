import os
from models.framework.train_test_framework import dual_enhance_sample_compare_UQ
import torch 

# main  
def NSvisc_samplebase_compareUQ_train_test(modelConfig):
    if os.path.exists(f"{modelConfig['MODEL_PATH']}/log.txt"):
        os.remove(f"{modelConfig['MODEL_PATH']}/log.txt")
    dual_enhance = dual_enhance_sample_compare_UQ(modelConfig)
    for ep in range(modelConfig['epochs']):   
        dual_enhance.epoch += 1
        dual_enhance.main_model_train(EPOCH=1)
        dual_enhance.UQ_model_train(epoch=1)
        dual_enhance.dual_enhance()
        
# dual_enhance_test without main model train
def dual_enhance_test(modelConfig):
    if os.path.exists(f"{modelConfig['MODEL_PATH']}/log.txt"):
        os.remove(f"{modelConfig['MODEL_PATH']}/log.txt")
    dual_enhance = dual_enhance_sample_compare_UQ(modelConfig)
    for ep in range(modelConfig['epochs']):   
        dual_enhance.epoch += 1
        dual_enhance.main_model_train(EPOCH=1)
        dual_enhance.UQ_model_train(epoch=1)
        dual_enhance.main_model_evaluate()
        dual_enhance.dual_enhance()
      
# main_model_evaluate function test:
def main_model_evaluate_function_test(modelConfig):
    if os.path.exists(f"{modelConfig['MODEL_PATH']}/log.txt"):
        os.remove(f"{modelConfig['MODEL_PATH']}/log.txt")
    dual_enhance = dual_enhance_sample_compare_UQ(modelConfig)
    for ep in range(modelConfig['epochs']):   
        dual_enhance.epoch += 1
        dual_enhance.main_model_train(EPOCH=1)
        dual_enhance.main_model_evaluate()
        dual_enhance.UQ_model_train(epoch=1)

# UQ_acc test
def UQ_acc_test(modelConfig):
    if os.path.exists(f"{modelConfig['MODEL_PATH']}/log.txt"):
        os.remove(f"{modelConfig['MODEL_PATH']}/log.txt")
    dual_enhance = dual_enhance_sample_compare_UQ(modelConfig)
    for ep in range(modelConfig['epochs']):   
        dual_enhance.epoch += 1
        dual_enhance.main_model_train(epoch=1)
        dual_enhance.UQ_model_train(EPOCH=2)
        dual_enhance.UQ_acc_plot(dual_enhance.epoch)
        dual_enhance.dual_enhance()

# multi plot test
def multi_plot_test(modelConfig):
    if os.path.exists(f"{modelConfig['MODEL_PATH']}/log.txt"):
        os.remove(f"{modelConfig['MODEL_PATH']}/log.txt")
    dual_enhance = dual_enhance_sample_compare_UQ(modelConfig)
    for ep in range(modelConfig['epochs']):   
        dual_enhance.epoch += 1
        dual_enhance.main_model_train(epoch=1)
        dual_enhance.UQ_model_train(EPOCH=2)
        dual_enhance.UQ_acc_plot(dual_enhance.epoch)
        dual_enhance.multi_pre_plot(step=10, epoch=dual_enhance.epoch)
        dual_enhance.dual_enhance()

# load a scale model and then dual enhance
def dual_enhance_scale_model(modelConfig):
    if os.path.exists(f"{modelConfig['MODEL_PATH']}/log.txt"):
        os.remove(f"{modelConfig['MODEL_PATH']}/log.txt")
    dual_enhance = dual_enhance_sample_compare_UQ(modelConfig)
    dual_enhance.datadriven_model.model = torch.load(modelConfig['path'])
    for ep in range(modelConfig['epochs']):   
        dual_enhance.epoch += 1
        dual_enhance.datadriven_model_train(epoch=1)
        dual_enhance.UQ_model_train(EPOCH=2)
        dual_enhance.UQ_acc_plot(dual_enhance.epoch)
        dual_enhance.main_model_evaluate()
        dual_enhance.dual_enhance()

# dual effect test
def dual_effect_test(modelConfig):
    if os.path.exists(f"{modelConfig['MODEL_PATH']}/log.txt"):
        os.remove(f"{modelConfig['MODEL_PATH']}/log.txt")
    dual_enhance = dual_enhance_sample_compare_UQ(modelConfig)
    for ep in range(modelConfig['epochs']):   
        dual_enhance.epoch += 1
        dual_enhance.main_model_train(epoch=1)
        dual_enhance.UQ_model_train(EPOCH=1)
        dual_enhance.UQ_acc_plot(dual_enhance.epoch)
        dual_enhance.multi_pre_plot(step=10, epoch=dual_enhance.epoch)
        dual_enhance.dual_enhance()
        

def UQ_test(modelConfig):
    if os.path.exists(f"{modelConfig['MODEL_PATH']}/log.txt"):
        os.remove(f"{modelConfig['MODEL_PATH']}/log.txt")
    dual_enhance = dual_enhance_sample_compare_UQ(modelConfig)
    # dual_enhance.datadriven_model.model = torch.load(modelConfig['path_d'])
    # dual_enhance.physics_model.model = torch.load(modelConfig['path_p'])
    # print("model loaded!")
    for ep in range(modelConfig['epochs']):   
        dual_enhance.epoch += 1
        dual_enhance.main_model_train(epoch=1)
        dual_enhance.UQ_model_train(EPOCH=1)
        dual_enhance.UQ_acc_plot(dual_enhance.epoch)
        dual_enhance.multi_pre_plot(step=10, epoch=dual_enhance.epoch)
        dual_enhance.dual_enhance()
        
        