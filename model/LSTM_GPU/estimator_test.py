
import tensorflow as tf
import numpy as np
import pandas as pd

HIDDEN_SIZE = 500
NUM_LAYERS = 2
TIMESTEPS = 1
TRAINING_STEPS = 10000
BATCH_SIZE = 1000
INPUT_SIZE = 9
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.9
NUM_EXAMPLES = 50000
MODEL_SAVE_PATH = "model_saved/"
MODEL_NAME = "model.ckpt"
LSTM_KEEP_PROB = 0.9
dropout_keep_prob=0.9

csv_column_names=['pitch_Atech_hub_temp_1',
                           'pitch_Atech_cabinet_temp_1',
                           'pitch_Atech_motor_current_1',
                           'pitch_position_1', 'converter_power', 'wind_speed', 'rotor_speed',
                           'pitch_Atech_motor_temp_1',
                           'pitch_Atech_converter_temp_1',
                            'pitch_Atech_capacitor_temp_1'
                           ]
species=['pitch_Atech_capacitor_temp_1']
from model.get_data_path import get_train_data_path
datafile=get_train_data_path('wfzc','A2','cap_temp_1','model_params_v3')
import pandas as pd
train=pd.read_csv(datafile,names=csv_column_names,header=0,index_col=0)
train=train.astype(dtype='float32')
print(train.head())
train_y=train.pop('pitch_Atech_capacitor_temp_1')
print(train_y.head())


def input_fn( batch_size=BATCH_SIZE):
    """An input function for training or evaluating"""
    # 将输入转换为数据集。
    dataset = tf.data.Dataset.from_tensor_slices((dict(train), train_y))

    # 如果在训练模式下混淆并重复数据。
    # if training:
    #     #     dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

def lstm_model(x,dropout_keep_prob):
    lstm_cells = [
        tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
            output_keep_prob=dropout_keep_prob)
        for _ in range(NUM_LAYERS)]
    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    print(np.shape(x))
    print("cell_created")
    x=tf.reshape(x,[-1,TIMESTEPS,INPUT_SIZE])
    print(np.shape(x))
    outputs,_=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    outputs=outputs[:,-1,:]
    predictions01 = tf.contrib.layers.fully_connected(outputs, 100,activation_fn= tf.nn.relu)
    predictions = tf.contrib.layers.fully_connected(predictions01, 1, activation_fn=None)
    return predictions

feature_columns=[]
for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key,dtype=tf.float32))
params={'learning_rate':1e-4,'feature_columns':feature_columns}
def model_fn(features,labels,mode,params):
    x=tf.feature_column.input_layer(features=features,feature_columns=params['feature_columns'])
    lstm_cells = [
        tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
            output_keep_prob=dropout_keep_prob,dtype=tf.float32)
        for _ in range(NUM_LAYERS)]
    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    print(np.shape(x))
    print("cell_created")
    x = tf.reshape(x, [-1, TIMESTEPS, INPUT_SIZE])
    print(np.shape(x))
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    outputs = outputs[:, -1, :]
    predictions01 = tf.contrib.layers.fully_connected(outputs, 100, activation_fn=tf.nn.relu)
    pred= tf.contrib.layers.fully_connected(predictions01, 1, activation_fn=None)
    if mode==tf.estimator.ModeKeys.PREDICT:
        spec=tf.estimator.EstimatorSpec(mode=mode,predictions=pred)
    else:
        loss_mse=tf.losses.mean_squared_error(labels=tf.reshape(labels,[-1]),predictions=tf.reshape(pred,[-1]))
        optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_BASE)
        train_op=optimizer.minimize(loss_mse,global_step=tf.train.get_global_step())
        metrics={'error':tf.metrics.mean_squared_error(labels=labels,predictions=pred)}
        spec=tf.estimator.EstimatorSpec(mode=mode,loss=loss_mse,train_op=train_op,eval_metric_ops=metrics)
    return spec

print(feature_columns)

model=tf.estimator.Estimator(model_fn=model_fn,params=params,model_dir='./lstm_estmator')
model.train(input_fn=input_fn,steps=1000)
# regression=tf.estimator.DNNRegressor(feature_columns=feature_columns,hidden_units=[30,10])
# regression.train(input_fn=lambda :input_fn(train,train_y,training=True),steps=5000)
# eval_result=regression.evaluate(input_fn=lambda :input_fn(train,train_y,training=False))
