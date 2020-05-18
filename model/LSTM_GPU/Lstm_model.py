import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.show_save_result import ShowAndSave
from model.get_data_path import get_train_data_path,get_test_data_path
import os

cur_path = os.path.abspath(os.path.dirname(__file__))



class Lstm_Model(ShowAndSave):
    def __init__(self,job_name=None,fc=None,fj=None,model_kind=None):
        super().__init__()

        self.columns=['pitch_Atech_hub_temp_1', 'pitch_Atech_cabinet_temp_1',
          'pitch_position_1', 'wind_speed', 'rotor_speed',
         'pitch_Atech_capacitor_temp_1']
        self.job_name = job_name + '_' + fc + '_' + fj
        self.model_folder_name = fj + '_' + model_kind
        self.model_name = fj
        self.fc = fc
        self.fj = fj
        self.model_kind = model_kind
        self.cur_path = cur_path
        self.init_param()
        self.fj_model_kind = fj + '_' + model_kind
        self.field_default=[0,0,0,0,0,0]
        self.train_file_path=get_train_data_path(fc=fc,fj=fj,model_kind=model_kind)
        print(self.train_file_path)
        self.batch_size=100

    def input_fn(self):
        def _parse_line(line):
            field_default = [0, 0, 0, 0, 0, 0]
            items=tf.decode_csv(line,field_default)
            features=items[:-1]
            label=items[-1]
            features=tf.cast(features,tf.float32)
            label=tf.cast(label,tf.float32)
            # features=dict(zip(self.columns,features))
            # label=features.pop('pitch_Atech_capacitor_temp_1')
            #label=features[-1]
            # features={'features':features}
            # label={'label':label}
            return features,label
        #ds=tf.data.TextLineDataset(self.train_file_path).skip(1)
        df=pd.read_csv(self.train_file_path,index_col=0,header=None,encoding='utf-8',low_memory=False)
        # ds=df.values
        # ds=tf.data.Dataset.from_tensor_slices(ds)

        # ds=ds.batch(self.batch_size)
        # iterator=ds.make_one_shot_iterator()
        # features=iterator.get_next()
        # with tf.Session() as sess:rrrr
        #     for i in range(2):
        #         print(sess.run([features]))
        #return features,label
        df = df.values
        ds = tf.data.Dataset.from_tensor_slices(df)
        #ds = ds.map(_parse_line)
        # ds = tf.data.TextLineDataset(
        #     '/home/HFData_PitchSystem_LSTM_Model/data/mergedData_new/wfzc_A2_cap_temp_1/wfzc_A2_cap_temp_1.csv')
        ds = ds.batch(self.batch_size)
        iterator = ds.make_one_shot_iterator()
        one_element= iterator.get_next()
        with tf.Session() as sess:
            for i in range(2):
                print(sess.run([one_element]))

    def get_feature_columns(self):
        feature_columns=[tf.feature_column.numeric_column(name) for name in self.columns[:-1]]
        return feature_columns

    def get_model(self):
        features_columns=self.get_feature_columns()
        regressor=tf.estimator.Estimator(model_fn=self.my_model_fn,model_dir=self.model_path,params={'feature_columns':features_columns,'hidden_units':[10,10]})

    def my_model_fn(self,features,labels,mode,params):
        input_net=tf.feature_column.input_layer(features,params['feature_columns'])
        for units in [10,10]:
            input_net=tf.layers.dense(input_net,units,activation=tf.nn.relu)
        logits=tf.layers.dense(input_net,1,activation=None)
        if mode==tf.estimator.ModeKeys.PREDICT:
            predictions={
                'class_ids':logits[:,tf.newaxis],
                'logits':logits

            }
            return tf.estimator.EstimatorSpec(mode,predictions=predictions)
        loss=tf.losses.mean_squared_error(labels=labels,logits=logits)
        accuracy=tf.metrics.accuracy(labels=labels,predictions=logits,name='acc_op')
        metrics={'accuracy':accuracy}
        if mode==tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)

        assert mode==tf.estimator.ModeKeys.TRAIN
        optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
        train_op=optimizer.minimize(loss)
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)

    def train_model(self):
        train_op=self.my_model_fn()

    def get_dataset(self):
        df=pd.read_csv('/home/HFData_PitchSystem_LSTM_Model/data/mergedData_new/wfzc_A2_cap_temp_1/wfzc_A2_cap_temp_1.csv',index_col=0)
        df=df.values
        ds=tf.data.Dataset.from_tensor_slices(df)
        # ds = tf.data.TextLineDataset(
        #     '/home/HFData_PitchSystem_LSTM_Model/data/mergedData_new/wfzc_A2_cap_temp_1/wfzc_A2_cap_temp_1.csv')
        ds=ds.batch(self.batch_size)
        iterator=ds.make_one_shot_iterator()
        one_element=iterator.get_next()
        with tf.Session() as sess:
            for i in range(2):
                print(sess.run([one_element]))


lm=Lstm_Model(job_name='lstm_model',fc='wfzc',fj='A2',model_kind='cap_temp_1')
lm.input_fn()
