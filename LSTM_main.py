from model.LSTM.LSTM_process import LSTM_Model
lstm = LSTM_Model(job_name='LSTM_Model', fc='wfzc',fj='A2',params_kind='model_params_v3',model_kind='cap_temp_1')
lstm.train_test_lstm()