import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def dd_model_postprocesse(model_dict, data_dict, x, y_solver, y, modelConfig, state):
    if state == 'train':
        out = model_dict['model'](x).view(modelConfig['batch_size'], modelConfig['S'], modelConfig['S'], modelConfig['T'])
        
        y = data_dict['y_normalizer_cuda'].decode(y)
        out = data_dict['y_normalizer_cuda'].decode(out)
        return out, y_solver, y
    
    elif state == 'test':
        out = model_dict['model'](x).view(modelConfig['batch_size'], modelConfig['S'], modelConfig['S'], modelConfig['T'])
        
        y = data_dict['y_normalizer_cuda'].decode(y)
        out = data_dict['y_normalizer_cuda'].decode(out)
        
        test_l20 = model_dict['myloss'](out.view(modelConfig['batch_size'], -1), y.view(modelConfig['batch_size'], -1)).item()
        return out, y_solver, y, [test_l20]

def p_model_postprocesse(model_dict, data_dict, x, y_solver, y, modelConfig, state):
    if state == 'train':
        out = model_dict['model'](x, y_solver).view(modelConfig['batch_size'], modelConfig['S'], modelConfig['S'], modelConfig['T'])
        out = out + y_solver.view(modelConfig['batch_size'], modelConfig['S'], modelConfig['S'], modelConfig['T'])
        
        y = data_dict['y_normalizer_cuda'].decode(y)
        y_solver = data_dict['y_normalizer_cuda'].decode(y_solver)
        out = data_dict['y_normalizer_cuda'].decode(out)
        return out, y_solver, y
    
    elif state == 'test':
        out = model_dict['model'](x, y_solver).view(modelConfig['batch_size'], modelConfig['S'], modelConfig['S'], modelConfig['T'])
        out = out + y_solver.view(modelConfig['batch_size'], modelConfig['S'], modelConfig['S'], modelConfig['T'])
        
        y = data_dict['y_normalizer_cuda'].decode(y)
        y_solver = data_dict['y_normalizer_cuda'].decode(y_solver)
        out = data_dict['y_normalizer_cuda'].decode(out)
        
        test_l20 = model_dict['myloss'](out.view(modelConfig['batch_size'], -1), y.view(modelConfig['batch_size'], -1)).item()
        test_l2_solver0 = model_dict['myloss'](y_solver.view(y.shape[0], -1), y.view(y.shape[0], -1)).item()
        return out, y_solver, y, [test_l20, test_l2_solver0]
    
    
