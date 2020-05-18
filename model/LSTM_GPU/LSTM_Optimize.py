import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from itertools import chain
import os
import numpy as np
import pandas as pd
from model.get_data_path import get_train_data_path, get_test_data_path
from model.LSTM.normalized_data import data_normalized
from sklearn.model_selection import train_test_split
from util.show_save_result import ShowAndSave

cur_path = os.path.abspath(os.path.dirname(__file__))
#datafile = get_train_data_path()


class LSTM_Model(ShowAndSave):

    def __init__(self, job_name='LSTM',fc='wfzc',fj='A2',model_kind='cap_temp_3', time_step=5, rnn_units=32,
                 batch_size=60, input_size=6, output_size=1, lr=0.0001,
                 train_steps=15000,datakind='train'):
        super().__init__()
        self.cur_path = cur_path
        self.job_name = job_name + '_' + fc + '_' + fj
        self.model_folder_name = fj + '_' + model_kind
        self.model_name = fj
        self.fc = fc
        self.fj = fj
        self.model_kind = model_kind
        self.init_param()
        self.fj_model_kind = fj + '_' + model_kind
        self.model_file_name = self.fj_model_kind + '.model'
        self.time_step = time_step
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.train_steps = train_steps
        self.data_kind=datakind

    def get_data(self):
        train_data = get_train_data_path(fc=self.fc,fj=self.fj,model_kind=self.model_kind)
        df = pd.read_csv(train_data, encoding='utf-8', index_col=0)
        data_set = df.iloc[:, :].values
        x_dataset = data_set[:, :-1]
        y_dataset = data_set[:, -1]
        self.x_train_mean = None
        self.x_train_std = None
        normalized_data = data_normalized(x_dataset)
        x_train, x_test, y_train, y_test = train_test_split(normalized_data, y_dataset, train_size=0.6)
        print(x_train[0], len(x_train))
        print(y_train[0], len(y_train))
        if self.data_kind == 'train':
            self.train_x_batch = []
            self.train_y_batch = []
            self.batch_index = []
            for i in range(len(x_train) - self.time_step - 1):
                if i % self.batch_size == 0:
                    self.batch_index.append(i)
                x = x_train[i:i + self.time_step]
                y = y_train[i:i + self.time_step]
                self.train_x_batch.append(x.tolist())
                self.train_y_batch.append(np.reshape(y, newshape=(self.time_step, self.output_size)))
            self.test_x_batch = []
            self.test_y_batch = []
            self.test_batch_index = []
            for i in range(len(x_test) - self.time_step - 1):
                if i % self.batch_size == 0:
                    self.test_batch_index.append(i)
                x = x_train[i:i + self.time_step]
                y = y_train[i:i + self.time_step]
                self.test_x_batch.append(x.tolist())
                self.test_y_batch.append(np.reshape(y, newshape=(self.time_step, self.output_size)))
        elif self.data_kind == 'fault_test':
            fault_test_data = get_test_data_path()
            falut_df = pd.read_csv(fault_test_data, encoding='utf-8', index_col=0)
            fault_data_set = falut_df.iloc[:, :].values
            fault_x_dataset = fault_data_set[:, :-1]
            fault_y_dataset = fault_data_set[:, -1]
            fault_normalized_data = data_normalized(fault_x_dataset)
            print(len(fault_normalized_data), len(fault_y_dataset))
            self.train_x_batch = []
            self.self.tmp_y_batch = []
            batch_index = []
            for i in range(len(fault_normalized_data) - self.time_step - 1):
                if i % self.batch_size == 0:
                    self.batch_index.append(i)
                x = fault_normalized_data[i:i + self.time_step]
                y = fault_y_dataset[i:i + self.time_step]
                self.train_x_batch.append(x.tolist())
                self.train_y_batch.append(np.reshape(y, newshape=(self.time_step, self.output_size)))

        # else:
        #     tmp_x_batch = []
        #     tmp_y_batch = []
        #     batch_index = []
        #     for i in range(len(x_train) - self.time_step - 1):
        #         if i % self.batch_size == 0:
        #             batch_index.append(i)
        #         x = x_train[i:i + self.time_step]
        #         y = y_train[i:i + self.time_step]
        #         tmp_x_batch.append(x.tolist())
        #         tmp_y_batch.append(np.reshape(y, newshape=(self.time_step, self.output_size)))
        #     return tmp_x_batch, tmp_y_batch, batch_index

    def model_fn(self,x,y,mode,params):
        pred=self.lstm(x)
        loss=tf.losses.mean_squared_error(labels=tf.reshape(y, [-1]),predictions=tf.reshape(pred,[-1]))
        trian_op=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        spec=tf.estimator.EstimatorSpec(mode=mode,loss=loss,trian_op=trian_op)
        return spec

    def init_lstm_params(self):
        self.weights = {
            'in': tf.Variable(tf.random_normal([self.input_size, self.rnn_units])),
            'out': tf.Variable(tf.random_normal([self.rnn_units, self.output_size]))
        }
        self.bias = {
            'in': tf.Variable(tf.constant(0.1, shape=[1, self.rnn_units])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }

    def lstm(self, x):
        batch_size = tf.shape(x)[0]
        time_step = tf.shape(x)[1]
        self.init_lstm_params()
        w_in = self.weights['in']
        b_in = self.bias['in']
        input = tf.reshape(x, [-1, self.input_size])
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, self.rnn_units])
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_units, reuse=tf.AUTO_REUSE)
        mcells = tf.nn.rnn_cell.MultiRNNCell([single_cell] * 2)
        init_state = mcells.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_state = tf.nn.dynamic_rnn(mcells, input_rnn, initial_state=init_state)
        output = tf.reshape(output_rnn, [-1, self.rnn_units])
        w_out = self.weights['out']
        b_out = self.bias['out']
        y = tf.matmul(output, w_out) + b_out
        # output =output_rnn[:,-1,:]
        # pred=fully_connected(inputs=output,num_outputs=self.batch_size*self.time_step,activation_fn=tf.nn.relu)
        # y=fully_connected(inputs=pred,num_outputs=1,activation_fn=None)
        return y, final_state

    def train_lstm(self):
        print(self.train_x_batch[0], len(self.train_x_batch))
        print(self.train_y_batch[0], len(self.train_y_batch))
        x_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.input_size])
        y_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.output_size])
        pred, _ = self.lstm(x_placeholder)
        loss_lstm = tf.reduce_sum(tf.square(tf.reshape(pred, [-1]) - tf.reshape(y_placeholder, [-1])))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss_lstm)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)
        print('train_path:' + self.model_path)
        with tf.Session() as sess:
            try:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            except:
                print('restore modle failue,starting trianing new model.')
            for step in range(self.train_steps):
                for i in range(len(self.batch_index) - 2):
                    _, _loss = sess.run([train_op, loss_lstm],
                                        feed_dict={
                                            x_placeholder: self.train_x_batch[self.batch_index[i]:self.batch_index[i + 1]],
                                            y_placeholder: self.train_y_batch[self.batch_index[i]:self.batch_index[i + 1]]})
                saver.save(sess, self.model_path + self.job_name + '_model')
                print('epo:' + str(step + 1) + '------' + '_loss:' + str(_loss))



    def test_lstm(self, delta_idx=50):
        print(self.test_x_batch[0], len(self.test_x_batch))
        print(self.test_y_batch[0], len(self.test_y_batch))
        x = tf.placeholder(tf.float32, [None, self.time_step, self.input_size])
        pred, _ = self.lstm(x)
        saver = tf.train.Saver(tf.global_variables(), reshape=True)

        print('test_path' + self.model_path)
        result = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            for i in range(len(self.test_x_batch)):
                res = sess.run(pred, feed_dict={x: [self.test_x_batch[i]]})
                # print(res)
                result.append(res[-1][0])
        true_y = []

        for e in self.test_y_batch:
            true_y.append(e[-1].tolist()[0])

        true_y = np.array(true_y)
        result = np.array(result)
        self.set_var(true_y, result)
        self.show_save_figure(detal_idx=delta_idx)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        info_dict = {'time_step': self.time_step, 'rnn_units': self.rnn_units, 'batch_size': self.batch_size,
                     'input_size': self.input_size, 'output_size': self.output_size, 'lr': self.lr,
                     'train_step': self.train_steps, 'data_mean': self.x_train_mean, 'data_std': self.x_train_std}
        self.save_result(true_mean=t_mean, pred_mean=p_mean, test_n=len(res), hyper_params=info_dict)

    def train_test_lstm(self):
        self.get_data()
        self.train_lstm()
        self.test_lstm()

    def get_estimator(self):
        params={'learning_rate':1e-4}
        model=tf.estimator.Estimator(model_fn=self.model_fn,params=params,model_dir=self.model_path)

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



lstm = LSTM_Model(job_name='LSTM_Model', fc='wfzc',fj='A2',model_kind='cap_temp_1')
lstm.get_dataset()

# print(tf.__version__)
