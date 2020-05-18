import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import pandas as pd
from model.get_data_path import get_train_data_path, get_test_data_path
from model.LSTM.normalized_data import data_normalized
from sklearn.model_selection import train_test_split
from util.show_save_result import ShowAndSave
import os

class GPU_LSTM(ShowAndSave):

    def __init__(self, job_name='LSTM', model_folder_name='418', model_name='1058'):
        super().__init__()
        self.job_name=job_name
        self.model_folder_name=model_folder_name
        self.model_name=model_name
        self.init_param()
        self.hidden_size=500
        self.num_layers=2
        self.time_step=10
        self.train_steps=10000
        self.batch_size=100
        self.input_size=6
        self.lr=0.001
        self.lrd=0.95
        self.n_sample=None
        self.log_path=self.single_model_path+'logs/'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_file_path=self.log_path+self.model_name
        if not os.path.exists(self.log_file_path):
            os.makedirs(self.log_file_path)

    def lstm(self,x,dropout_keep_prob):
        single_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)
        dropout_lstms=[tf.nn.rnn_cell.DropoutWrapper(single_cell,output_keep_prob=dropout_keep_prob) for _ in range(self.num_layers)]
        mcells=tf.nn.rnn_cell.MultiRNNCell(dropout_lstms)
        print('cell created.')
        output,final_state=tf.nn.dynamic_rnn(mcells,x,dtype=tf.float32)
        output=output[:,-1,:]
        pred=fully_connected(output,64,activation_fn=tf.nn.relu)
        resu=fully_connected(pred,1,activation_fn=None)
        return resu

    def get_data(self, datakind):
        self.data_kind = datakind
        train_data = get_train_data_path()

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
            tmp_x_batch = []
            tmp_y_batch = []
            batch_index = []
            for i in range(len(x_train) - self.time_step - 1):
                if i % self.batch_size == 0:
                    batch_index.append(i)
                x = x_train[i:i + self.time_step]
                y = y_train[i:i + self.time_step]
                tmp_x_batch.append(x.tolist())
                tmp_y_batch.append(np.reshape(y, newshape=(self.time_step, self.output_size)))

            return tmp_x_batch, tmp_y_batch, batch_index
        elif self.data_kind == 'fault_test':
            fault_test_data = get_test_data_path()
            falut_df = pd.read_csv(fault_test_data, encoding='utf-8', index_col=0)
            fault_data_set = falut_df.iloc[:, :].values
            fault_x_dataset = fault_data_set[:, :-1]
            fault_y_dataset = fault_data_set[:, -1]
            fault_normalized_data = data_normalized(fault_x_dataset)
            print(len(fault_normalized_data), len(fault_y_dataset))
            tmp_x_batch = []
            tmp_y_batch = []
            batch_index = []
            for i in range(len(fault_normalized_data) - self.time_step - 1):
                if i % self.batch_size == 0:
                    batch_index.append(i)
                x = fault_normalized_data[i:i + self.time_step]
                y = fault_y_dataset[i:i + self.time_step]
                tmp_x_batch.append(x.tolist())
                tmp_y_batch.append(np.reshape(y, newshape=(self.time_step, self.output_size)))
            return tmp_x_batch, tmp_y_batch, batch_index
        else:
            tmp_x_batch = []
            tmp_y_batch = []
            batch_index = []
            for i in range(len(x_train) - self.time_step - 1):
                if i % self.batch_size == 0:
                    batch_index.append(i)
                x = x_train[i:i + self.time_step]
                y = y_train[i:i + self.time_step]
                tmp_x_batch.append(x.tolist())
                tmp_y_batch.append(np.reshape(y, newshape=(self.time_step, self.output_size)))
            return tmp_x_batch, tmp_y_batch, batch_index

    def train_lstm(self,x_train,y_train,datakind='train'):
        self.data_kind=datakind
        global_step=tf.Variable(0,trainable=False)
        ds=tf.data.Dataset.from_tensor_slices((x_train,y_train))#错了。
        ds=ds.repeat().shuffle(1000000).batch(self.batch_size)
        x,y=ds.make_one_shot_iterator().get_next()
        pred=self.lstm(x,dropout_keep_prob=0.9)
        loss=tf.losses.mean_squared_error(labels=y,predictions=pred)
        tf.summary.scalar('loss',loss)
        lr=tf.train.exponential_decay(self.lr,global_step,self.n_sample/self.batch_size,self.lrd)
        train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,global_step=global_step)
        saver=tf.train.Saver()
        merged=tf.summary.merge_all()
        train_writer=tf.summary.FileWriter(self.log_file_path+'log',tf.get_default_graph())
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(self.train_steps):
                if i %100==0:
                    run_op=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata=tf.RunMetadata()
                    summary,_,loss_,step=sess.run([merged,train_op,loss,global_step],options=run_op,run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata,'step%02d' % i)
                    train_writer.add_summary(summary,i)
                    print("step:%s-----loss:%s" % (str(step),step(loss_)))
                    saver.save(sess,self.model_path)
                else:
                    summary,_,loss_,step=sess.run([merged,train_op,loss,global_step])
                    train_writer.add_summary(summary,i)
        train_writer.close()

    def test_lstm(self,datakind='test'):
        x_test,y_test=None,None
        ds=tf.data.Dataset.from_tensor_slices((x_test,y_test))
        ds=ds.batch(1)
        x,y=ds.make_one_shot_iterator().get_next()
        pred=self.lstm(x,dropout_keep_prob=1)
        pred_list=[]
        label=[]
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                print('checkpoint found.')
            else:
                print('checkpoint not found.')
            for i in range(len(y_test)):
                predget,yy=sess.run([pred,y])
                yy=np.array(yy).squeeze()
                pred_list.append(predget)

                