import pandas as pd
import lightgbm as lgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
import pandas as pd
import matplotlib.pyplot as plt
from model.get_data_path import get_filtered_data_path,get_test_data_path,get_train_data_path
from sklearn.model_selection import train_test_split
import os
from util.show_save_result import ShowAndSave


cur_path=os.path.abspath(os.path.dirname(__file__))



class LightGBM_Model(ShowAndSave):
    def __init__(self, params=None, jobname='lgbm_model', fc='wfzc',fj='A2',model_kind='cap_temp_1',params_kind=None,max_depth=30,n_estimator=1024,num_leaves=3):
        super().__init__()
        self.job_name=jobname+'_'+fc+'_'+fj+'_'+str(max_depth)+'_'+str(n_estimator)
        print(self.job_name)
        self.model_folder_name=fj+'_'+model_kind
        self.model_name=fj
        self.model_kind=model_kind
        self.fc=fc
        self.fj=fj
        self.model_kind=model_kind
        self.cur_path=cur_path
        self.init_param()
        self.params = params
        self.fj_model_kind=fj+'_'+model_kind
        self.model_file_name=self.fj_model_kind+'_lgbm.model'
        self.feature_importance_path=self.single_model_path+'feature_importance/'
        self.max_depth=max_depth
        self.n_estimator=n_estimator
        self.num_leaves=num_leaves
        self.params_kind=params_kind

    def train_lightgbm_model(self,data_kind='train'):
        self.data_kind=data_kind
        data_file=get_train_data_path(self.fc,self.fj,self.model_kind,self.params_kind)
        df=pd.read_csv(data_file,index_col=0,encoding='utf-8',low_memory=False)
        #print(df.columns)
        print('df shape:',df.shape)
        traindata = df.iloc[:, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)  # list
        print(len(x_train),len(x_test))
        print('training lightgbm model')
        model=LGBMRegressor(boosting_type='gbdt',num_leaves=self.num_leaves,n_estimators=self.n_estimator,max_depth=self.max_depth)
        model.fit(x_train,y_train)
        print(model.feature_importances_)
        print(model.best_score_)
        print(model.best_iteration_)
        joblib.dump(model,self.model_path+self.model_name)
        pred=model.predict(x_test)
        self.set_var(true_v=y_test, pred_v=pred)
        self.show_save_figure(detal_idx=8)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean, train_n=len(x_train), test_n=len(x_test))

    def test_model(self,data_kind='test'):
        self.data_kind=data_kind
        fault_test_file_path = get_test_data_path(self.fc, self.fj, self.model_kind,self.params_kind)
        df = pd.read_csv(fault_test_file_path, encoding='utf-8', index_col=0)
        data = df.iloc[:, :].values
        x = data[:, :-1]
        y = data[:, -1]
        raw_model = joblib.load(self.model_path + self.model_name)
        pred = raw_model.predict(x)
        self.true = y
        self.pred = pred
        self.show_save_figure(detal_idx=4)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean)

lgbm=LightGBM_Model(jobname='lgbm_model_motor',fc='wfzc',fj='A2',model_kind='motor_temp_2',params_kind='model_params_v2',num_leaves=512,n_estimator=1024,max_depth=256)
lgbm.train_lightgbm_model()
lgbm.test_model()
