import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd


class ShowAndSave:
    def __init__(self):
        self.fig_path = None
        self.pred = None
        self.true = None
        self.multi_model_path = None
        self.single_model_path = None
        self.result_path = None
        self.model_path = None
        self.model_kind=None
        self.job_name = None
        self.model_folder_name=None
        self.model_name=None
        self.params_file_path=None
        self.cur_path = None
        self.params=None
        self.data_kind=None
        # self.fault_data_test_result_path=None
        # self.fault_data_test_figure_path=None
        self.task_name=None
        self.true_pred_path=None
        #self.params_kind=None


    def init_param(self, ):
        self.time_now=time.strftime('%H%M')
        # parent_path=os.path.abspath(os.path.join(cur_path,'..'))
        self.multi_model_path = self.cur_path + '/' + self.job_name
        if not os.path.exists(self.multi_model_path):
            os.makedirs(self.multi_model_path)
        self.single_model_path = self.multi_model_path + '/'+ self.model_folder_name + '/'
        self.model_path = self.single_model_path + 'model_' + self.model_name + '/'
        #self.model_path = '/home/HFData_PitchSystme_LSTM_Model/model/LSTM/LSTM/LSTM_20200417/model_16/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # self.fault_data_test_result_path= self.single_model_path +'test_result_' + model_name + '/'
        # if not os.path.exists(self.fault_data_test_result_path):
        #     os.makedirs(self.fault_data_test_result_path)
        # self.fault_data_test_figure_path= self.single_model_path +'test_figure_' + model_name + '/'
        # if not os.path.exists(self.fault_data_test_figure_path):
        #     os.makedirs(self.fault_data_test_figure_path)
        self.params_file_path= self.single_model_path +'params_file_' + self.model_name +'_'+self.time_now+ '/'
        if not os.path.exists(self.params_file_path):
            os.makedirs(self.params_file_path)

    def mk_result_path(self):
        self.result_path = self.single_model_path + 'result_' + self.model_folder_name + '_' + self.data_kind + '/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def set_var(self,true_v,pred_v):
        self.true=true_v
        self.pred=pred_v

    def cal_error(self):
        # 均方误差
        mse = np.sum((self.pred - self.true) ** 2 / len(self.true))
        # 均方根误差
        rmse = mse ** 0.5
        # 平均绝对值误差
        mae = np.sum(np.absolute(self.pred - self.true)) / len(self.true)
        return mse, rmse, mae

    def show_save_figure(self, detal_idx=10):
        self.fig_path = self.single_model_path + 'figure_' + self.model_name + '_' + self.data_kind + '/'
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)
        self.error_path=self.single_model_path+'error_'+self.model_name+'_'+self.data_kind+'/'
        if not os.path.exists(self.error_path):
            os.makedirs(self.error_path)
        true_li = []
        error_li = []
        pred_li = []
        for i in range(len(self.true)):
            if i % detal_idx == 0:
                true_li.append(self.true[i])
                error_li.append(self.true[i]-self.pred[i])
                pred_li.append(self.pred[i])
        x = np.array(range(len(true_li)))
        # print('error_list',len(true_li))
        error_df=pd.DataFrame({'error':error_li})

        if os.path.exists(self.error_path+self.data_kind+'error.csv'):
            error_df.to_csv(self.error_path + self.data_kind +self.time_now+'_error.csv', encoding='utf-8')
        else:
            error_df.to_csv(self.error_path+self.data_kind+'error.csv',encoding='utf-8')
        plt.plot(x, true_li, color="green", label="true")
        plt.plot(x, pred_li, color="blue", label="pred")
        #plt.plot(x,after_rolled_error_df_list,label='error_after_rolled')
        # plt.show()
        plt.plot(x, error_li, color="red", label='error')  # 画图
        plt.legend(loc='upper left',bbox_to_anchor=(0.01,0.99))
        plt.title(self.job_name)
        if os.path.exists(self.fig_path + self.model_folder_name+'_'+self.model_kind+'_'+self.data_kind+'.png'):
            plt.savefig(self.fig_path + self.model_folder_name + '_' + self.model_kind + '_' + self.data_kind+'_'+self.time_now)
        else:
            plt.savefig(self.fig_path + self.model_folder_name+'_'+self.model_kind+'_'+self.data_kind)
        plt.show()
        plt.close()

    def show_rolling_fig(self,detal_idx=10):
        true_li = []
        error_li = []
        pred_li = []
        for i in range(len(self.true)):
            if i % detal_idx == 0:
                true_li.append(self.true[i])
                error_li.append(abs(self.true[i] - self.pred[i]))
                pred_li.append(self.pred[i])
        x = np.array(range(len(true_li)))
        error_df = pd.DataFrame({'abs_error': error_li})
        error_df.rolling(10).mean().plot()
        error_df.plot()
        true_df = pd.DataFrame({'true_rolling': true_li})
        true_df.rolling(10).mean().plot()
        true_df.plot()
        pred_df = pd.DataFrame({'pred_rolling': pred_li})
        pred_df.rolling(10).mean().plot()
        pred_df.plot()


    def save_result(self, true_mean=None, pred_mean=None,train_n=None,test_n=None,hyper_params=None):
        self.result_path = self.single_model_path + 'result_' + self.model_name + '_' + self.data_kind + '/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        mse, rmse, mae = self.cal_error()
        mape=self.mean_absolute_percentage_error()
        rsqu=self.r_square()
        data = {'true_mean':true_mean,'pred_mean':pred_mean,'mse': mse, 'rmse': rmse, 'mae': mae,'mape':mape,'R Square':rsqu,'params':self.params,'train_number':train_n,'test_number':test_n,'hyper_params':hyper_params}
        print(data)
        df = pd.DataFrame(list(data.items()))
        #result_path=self.result_path +self.model_folder_name+'_'+self.model_kind+'_'+self.data_kind+ '_result.csv'
        if os.path.exists(self.result_path +self.model_folder_name+'_'+self.model_kind+'_'+self.data_kind+ '_result.csv'):
            df.to_csv(self.result_path +self.model_folder_name+'_'+self.model_kind+'_'+self.data_kind+'_'+self.time_now+ '_result.csv',encoding='utf-8')
        else:
            df.to_csv(self.result_path +self.model_folder_name+'_'+self.model_kind+'_'+self.data_kind+ '_result.csv',encoding='utf-8',index=None,header=None)

    def cal_mean(self, input):
        mean_val=np.mean(input)
        return mean_val

    #平均相对误差绝对值，用于刻画预测值和真实值之间的偏差，越小越好
    def mean_absolute_percentage_error(self):
        y_true, y_pred = np.array(self.true), np.array(self.pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    #用于刻画预测值与真实值之间的拟合程度，越大越好
    def r_square(self):
        #sse:预测数据与原始数据对应点的误差的平方和
        #ssr:预测数据与原始数据均值之差的平方和
        #sst：原始数据与均值之差的平方和

        true_mean=self.cal_mean(self.true)
        sse=0
        sst=0
        for i in range(len(self.pred)):
            sse+=(self.pred[i]-self.true[i])**2
            sst+=(self.true[i]-true_mean)**2
        r_square=1-sse/sst
        return r_square

    def save_result_dataframe(self,true,pred):
        self.true_pred_path=self.single_model_path + 'actual_prediction_' + self.model_name + '_' + self.data_kind + '/'
        if not os.path.exists(self.true_pred_path):
            os.makedirs(self.true_pred_path)
        df=pd.DataFrame({'true':true,'pred':pred})
        if os.path.exists(self.true_pred_path+self.model_folder_name+'_'+self.model_kind+'_actual_prediction.csv'):
            df.to_csv(self.true_pred_path + self.model_folder_name + '_' + self.model_kind +'_'+self.time_now+ '_actual_prediction.csv',
                      encoding='utf-8')
        else:
            df.to_csv(self.true_pred_path+self.model_folder_name+'_'+self.model_kind+'_actual_prediction.csv',encoding='utf-8')
