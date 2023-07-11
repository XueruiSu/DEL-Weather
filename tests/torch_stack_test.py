import torch  
from configs.Climax_train_modelparam import dict_atmos_vars
print(len(dict_atmos_vars["era5"]))
# 创建一些张量  
tensor1 = torch.tensor([1, 2, 3])  
tensor2 = torch.tensor([4, 5, 6])  
tensor3 = torch.tensor([7, 8, 9])  
  
# 使用 torch.stack 将张量沿新维度堆叠在一起  
stacked_tensor = torch.stack((tensor1, tensor2, tensor3), dim=0)  
  
print(stacked_tensor)  

stacked_tensor_dim1 = torch.stack((tensor1, tensor2, tensor3), dim=1)  
print(stacked_tensor_dim1)  
# dim range: [-2, 1]

stacked_tensor_dim1 = torch.stack((tensor1, tensor2, tensor3), dim=2)  
print(stacked_tensor_dim1)  # error