
from data_preprocess.data_filter import GetGuZhangData
import time
model_list=['cap_temp_1','cap_temp_2','cap_temp_3','convt_temp_1','convt_temp_2','convt_temp_3','motor_temp_1','motor_temp_2','motor_temp_3']
for model_kind in model_list:
    from model.xgboost.xgboostprocess import XgboostModel
    gz = GetGuZhangData('wfzc','A2',model_kind=model_kind)
    gz.process_data()
    xgbm = XgboostModel(jobname='xgb_model',fc='wfzc',fj='A2',model_kind=model_kind)
    xgbm.train_test_model()
    time.sleep(2)