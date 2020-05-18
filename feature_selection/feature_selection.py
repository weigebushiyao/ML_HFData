from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from util.data_load import train_test_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import multiprocessing
from model.get_data_path import get_train_data_path
import time

n_cpu=multiprocessing.cpu_count()
print(n_cpu)
cur_path=os.path.abspath(os.path.dirname(__file__))
parent_path=os.path.abspath(os.path.join(cur_path,'..'))

class FeatureSelection():
    def __init__(self,fc=None,fj=None,model_kind=None):
        self.fc=fc
        self.fj=fj
        self.model_kind=model_kind
        self.result_path=cur_path+'/'+'feature_selection_result'+'/'+fc+'_'+fj+'_'+model_kind+'/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.get_data()
        self.time_now=time.strftime('%d%H%M')

    def cal_Accuracy(self,pred, test_y,flag='reg'):
        if flag=='cls':
            count = 0
            pred_cnt = len(pred)
            for i in range(pred_cnt):
                if pred[i] == test_y[i]:
                    count += 1
            return round(count / pred_cnt, 2)
        else:
            error=np.array(pred)-np.array(test_y)
            return error

    def get_data(self):
        data_file = get_train_data_path(self.fc, self.fj, self.model_kind)
        df = pd.read_csv(data_file, encoding='utf-8', index_col=0, low_memory=False)
        print(df.shape)
        traindata = df.iloc[:, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, train_size=0.6)  # list

    def plot_result(self,true_y,pred_y,fig_name,detal_idx=2):
        true_li = []
        error_li = []
        pred_li = []
        for i in range(len(true_y)):
            if i % detal_idx == 0:
                true_li.append(true_y[i])
                error_li.append(true_y[i] - pred_y[i])
                pred_li.append(pred_y[i])
        x = np.array(range(len(true_li)))
        # print('error_list',len(true_li))
        error_df = pd.DataFrame({'error': error_li})
        error_path=self.result_path + 'error.csv'
        if os.path.exists(error_path):
            error_df.to_csv(self.result_path+self.time_now + '_error.csv', encoding='utf-8')
        else:
            error_df.to_csv(error_path, encoding='utf-8')
        plt.plot(x, true_li, color="green", label="true")
        plt.plot(x, pred_li, color="red", label="pred")
        plt.plot(x, error_li, color="blue", label='error')  # 画图
        plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
        plt.title(self.fc+'_'+self.fj+'_'+self.model_kind)
        if os.path.exists(self.result_path+fig_name):
            plt.savefig(self.result_path + fig_name+'_'+self.time_now)
        plt.savefig(self.result_path+fig_name)

    # 使用随机森林对特征进行选择,并传递重要性阈值
    def rank_feature_importance(self,thres):
        trees = 250
        max_feat = np.shape(self.train_x)[1]
        max_depth = 30
        min_sample = 2
        rfr = RandomForestRegressor(n_estimators=trees, max_features=max_feat, max_depth=max_depth,
                                    min_samples_split=min_sample, random_state=0, n_jobs=-1)
        import time
        start_time = time.time()
        rfr.fit(self.train_x, self.train_y)
        end_time = time.time()
        first_time = round(end_time - start_time, 2)
        data_shape = np.shape(np.array(self.train_x))
        pre = rfr.predict(self.test_x)
        self.plot_result(self.test_y,pred_y=pre,fig_name='before_selected')
        sfm = SelectFromModel(rfr, threshold=thres)
        sfm.fit(self.train_x, self.train_y)
        # new_train_x = sfm.transform(self.train_x)
        # new_test_x = sfm.transform(self.test_x)
        print(rfr.feature_importances_)
        print(data_shape)
        print(first_time)
        # new_data_shape = np.shape(new_train_x)
        # start_time2 = time.time()
        # rfr.fit(new_train_x, self.train_y)
        # end_time2 = time.time()
        # pre2 = rfr.predict(new_test_x)
        # self.plot_result(self.test_y,pre2,'after_selected')
        # acc2 = self.cal_Accuracy(pre2, self.test_y)
        # second_time = round(end_time2 - start_time2, 2)
        # print(new_data_shape)
        # print(second_time)
        # print(acc2)
        result_txt = {'original_data_shape': data_shape, '1st_train_time': first_time,
                      'feature_importance':rfr.feature_importances_}
        f = open(self.result_path + '/' + 'selected_feature.csv', 'w', newline='')
        for k, v in result_txt.items():
            f.write(k + ': ' + str(v) + '\n')
        f.close()


    # 基于递归特征选择方法
    def select_feature_RFE(self):
        model = LogisticRegression()
        n= np.shape(self.train_x)[1]
        rfe = RFE(model, n_features_to_select=n)
        rfe.fit(self.train_x, self.train_y)
        print(rfe.n_features_)
        print(rfe.support_)
        print(rfe.ranking_)

    # select_feature_RFE(12)

    def select_feature_ExtratTree(self):
        model = ExtraTreesRegressor()
        model.fit(self.train_x, self.train_y)
        fi=model.feature_importances_
        print(fi)
        df=pd.DataFrame({'feature_importance':fi})
        df.to_csv(self.result_path+'feature_importance.csv')

    # select_feature_ExtratTree()

fs=FeatureSelection(fc='wfzc',fj='A2',model_kind='cap_temp_1')
fs.rank_feature_importance(0.6)
#fs.select_feature_RFE()
fs.select_feature_ExtratTree()