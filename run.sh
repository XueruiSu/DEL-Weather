python main.py --model_s FNO_sample --visc 0.0008 --docu difvisc
python main.py --model_s FNO --visc 0.0008 --docu difvisc

python main.py --model_s FNO_sample --visc 0.001 --docu difvisc
python main.py --model_s FNO --visc 0.0008 --docu difvisc
# 0529跑的两个实验：一个是FNO_subnet UQ都改成在测试集上测试visc=0.001,use_y，
# 一个是(黄的那个)visc=0.0008，直接真的dual，但是PTD不多，所以dd的线相当于纯数据驱动。

# dual test:
python main.py --model_s FNO_sample --visc 0.001 --docu dualtest --start_dual_epoch 1 --use_y


# cd command:
cd /blob/xueruisu/FNO_base_UQ_for_scale_model/DEL-Reform
source activate
conda activate pytorch-py3.7
# 0530 needed lab:
# no_dual
CUDA_VISIBLE_DEVICES=0 python main.py --model_s FNO_sample --visc 0.0008 --docu no_dual --start_dual_epoch 700 &
# dual:
CUDA_VISIBLE_DEVICES=1 python main.py --model_s FNO_sample --visc 0.0008 --docu dual --start_dual_epoch 300 &
# dual_fake
CUDA_VISIBLE_DEVICES=2 python main.py --model_s FNO_sample --visc 0.0008 --docu dual_fake --start_dual_epoch 300 --use_y &
# UQ lab:
# mse:
CUDA_VISIBLE_DEVICES=0 python main.py --model_s FNO_sample --visc 0.0008 --docu MSE_subnet --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 0 --UQ_Rank 0 &
CUDA_VISIBLE_DEVICES=1 python main.py --model_s FNO --visc 0.0008 --docu MSE_mean --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 0 --UQ_Rank 0 &
# RE
CUDA_VISIBLE_DEVICES=2 python main.py --model_s FNO_sample --visc 0.0008 --docu RE_subnet --start_dual_epoch 1 --UQ_mse 0 --UQ_l2 1 --UQ_Rank 0 &
CUDA_VISIBLE_DEVICES=3 python main.py --model_s FNO --visc 0.0008 --docu RE_mean --start_dual_epoch 1 --UQ_mse 0 --UQ_l2 1 --UQ_Rank 0 &
# Rankloss
CUDA_VISIBLE_DEVICES=0 python main.py --model_s FNO_sample --visc 0.0008 --docu Rankloss_subnet --start_dual_epoch 1 --UQ_mse 0 --UQ_l2 0 --UQ_Rank 1 &
CUDA_VISIBLE_DEVICES=1 python main.py --model_s FNO --visc 0.0008 --docu Rankloss_mean --start_dual_epoch 1 --UQ_mse 0 --UQ_l2 0 --UQ_Rank 1 &


# in my GCR device:
python main.py --model_s FNO_sample --visc 0.0008 --docu MSE_subnet --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 0 --UQ_Rank 0
python main.py --model_s FNO --visc 0.0008 --docu MSE_mean --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 0 --UQ_Rank 0
# RE
python main.py --model_s FNO_sample --visc 0.0008 --docu RE_subnet --start_dual_epoch 1 --UQ_mse 0 --UQ_l2 1 --UQ_Rank 0
python main.py --model_s FNO --visc 0.0008 --docu RE_mean --start_dual_epoch 1 --UQ_mse 0 --UQ_l2 1 --UQ_Rank 0
# Rankloss
python main.py --model_s FNO_sample --visc 0.0008 --docu Rankloss_subnet --start_dual_epoch 1 --UQ_mse 0 --UQ_l2 0 --UQ_Rank 1
python main.py --model_s FNO --visc 0.0008 --docu Rankloss_mean --start_dual_epoch 1 --UQ_mse 0 --UQ_l2 0 --UQ_Rank 1


# mean111
python main.py --model_s FNO --visc 0.0008 --docu FNO_mean111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# subnet111
python main.py --model_s FNO_sample --visc 0.0008 --docu FNO_sample_subnet111 --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# mainmodel_subnet111_
python main.py --model_s FNO_sample_main --visc 0.0008 --docu main_FNO_sample_main111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# RE_mainmodel_subnet111_
python main.py --model_s FNO_sample_RE --visc 0.0008 --docu main_FNO_sample_RE111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1

# mainmodel_subnet111_
python main.py --model_s sample_main --visc 0.0008 --docu main_sample_main111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# RE_mainmodel_subnet111_
python main.py --model_s sample_RE --visc 0.0008 --docu main_sample_RE111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1





