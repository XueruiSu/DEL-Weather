import yaml  
from pytorch_lightning import Trainer  
# from pytorch_lightning.utilities import instantiate 
  
# 读取 YAML 文件并解析配置  
with open("./configs/pretrain_climnet_base.yaml", "r") as f:  
    config = yaml.safe_load(f)  
  
# 从配置中获取 Trainer 参数  
trainer_config = config["trainer"]  
  
# 处理 logger 和 callbacks  
# trainer_config["logger"] = instantiate(trainer_config["logger"])  
# trainer_config["callbacks"] = [instantiate(cb) for cb in trainer_config["callbacks"]]  
print(trainer_config)
# 创建 Trainer 实例  
trainer = Trainer(**trainer_config)  
