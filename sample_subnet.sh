# 0530:
# mainmodel_subnet111_
python main.py --model_s FNO_sample_main --visc 0.0008 --docu main_FNO_sample_main111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# RE_mainmodel_subnet111_
python main.py --model_s FNO_sample_RE --visc 0.0008 --docu main_FNO_sample_RE111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# mainmodel_subnet111 without FNO
python main.py --model_s sample_main --visc 0.0008 --docu main_sample_main111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# RE_mainmodel_subnet111 without FNO
python main.py --model_s sample_RE --visc 0.0008 --docu main_sample_RE111_ --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1

# 0605:
# dual effection test:
python main.py --model_s FNO_sample_RE --visc 0.001 --docu dual_effect_test --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# Use y as target
python main.py --model_s FNO_sample_RE --visc 0.001 --docu dual_effect_test --use_y --start_dual_epoch 1 --UQ_mse 1 --UQ_l2 1 --UQ_Rank 1
# pixel NLL test
python main.py --model_s pixel_NLL --visc 0.0008 --docu pixel_NLL_test --use_y --start_dual_epoch 1
# adv generation test:
python main.py --model_s FNO_sample_RE --visc 0.0008 --docu adv_generation_test --use_y --start_dual_epoch 1 --adv_generate
# Ensemble UQ test
python main.py --model_s Ensemble --visc 0.0008 --docu Ensemble_UQ_test --start_dual_epoch 1 

