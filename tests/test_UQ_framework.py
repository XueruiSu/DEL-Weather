from models.framework.train_test_framework import UQ_Dual_Weather_Climai

UQ_Dual_Weather_Climai_obj = UQ_Dual_Weather_Climai()


print("test of UQ_data_iterative_generation_Functest")
[single_inputs, atmos_inputs, 
 single_outputs, atmos_outputs, 
 lead_times, metadata, 
 single_pre_1, atmos_pre_1, 
 single_pre_2, atmos_pre_2] = UQ_Dual_Weather_Climai_obj.UQ_data_iterative_generation_Functest()
print("inputs:")
print("single_inputs", single_inputs.shape)
for k in atmos_inputs:
    print("atmos_inputs", k, atmos_inputs[k].shape)
print("lead_times", lead_times.shape)
print("metadata lat", metadata.lat.shape)
print("metadata lon", metadata.lon.shape)
print("outputs:")
print("single_outputs", single_outputs.shape)
for k in atmos_outputs:
    print("atmos_outputs", k, atmos_outputs[k].shape)
print("single_pre_1", single_pre_1.shape)
for k in atmos_pre_1:
    print("atmos_pre_1", k, atmos_pre_1[k].shape)
print("single_pre_2", single_pre_2.shape)
for k in atmos_pre_2:
    print("atmos_pre_2", k, atmos_pre_2[k].shape)