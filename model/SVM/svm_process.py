#-*-coding:utf-8-*-
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from model.get_data_path import get_filtered_data_path,get_test_data_path,get_train_data_path
from sklearn.model_selection import train_test_split
import os
from util.show_save_result import ShowAndSave


cur_path=os.path.abspath(os.path.dirname(__file__))



class XgboostModel(ShowAndSave):
    def __init__(self, params=None, jobname='svm_model', fc='wfzc',fj='A2',model_kind='cap_temp_1',max_depth=64,n_estimator=512,min_child_weight=3):
        super().__init__()
        self.job_name=jobname+'_'+fc+'_'+fj+'_'+str(max_depth)+'_'+str(n_estimator)
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
        self.model_file_name=self.fj_model_kind+'_svm.model'
        self.feature_importance_path=self.single_model_path+'feature_importance/'
        self.max_depth=max_depth
        self.n_estimator=n_estimator
        self.min_child_weight=min_child_weight

    def svm_model(self,data_kind='train_data'):
        self.data_kind=data_kind
        data_file=get_train_data_path(self.fc,self.fj,self.model_kind)
        df = pd.read_csv(data_file, encoding='utf-8', index_col=0,low_memory=False)
        print(df.shape)
        traindata = df.iloc[:, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)  # list
        print('when training,train data number:',len(y_train))
        print('when training,test data number:',len(y_test))
        print('training model:',self.model_kind)
        raw_model=SVR()
        parameters = [
            {
                'C': [3, 7, 9, 15,  19],
                'gamma': [  0.001,  1, 10, ],
                'kernel': ['rbf']
            }
        ]
        gsc=GridSearchCV(raw_model,parameters,cv=3)
        gsc.fit(x_train,y_train)

        raw_model.fit(x_train, y_train)
        print('svm best parameters:', gsc.best_params_)
        print(self.model_path)
        joblib.dump(gsc,self.model_path+self.model_name)
        pred = raw_model.predict(x_test)

        self.save_result_dataframe(y_test,pred,)
        self.set_var(true_v=y_test,pred_v=pred)
        self.show_save_figure(detal_idx=4)
        t_mean=self.cal_mean(self.true)
        p_mean=self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean,train_n=len(x_train),test_n=len(x_test))

    def test_model(self,data_kind='fault_data',delta_idx=2):
        self.data_kind=data_kind
        fault_test_file_path=get_test_data_path(self.fc,self.fj,self.model_kind)
        df=pd.read_csv(fault_test_file_path,encoding='utf-8',index_col=0)
        data=df.iloc[:,:].values
        x=data[:,:-1]
        print(x[0])
        y=data[:,-1]

        svr=joblib.load(self.model_path+self.model_name)
        print(self.model_path+self.model_file_name)

        pred=svr.predict(x)
        self.save_result_dataframe(y,pred)
        self.set_var(true_v=y,pred_v=pred)
        #self.show_rolling_fig()
        self.show_save_figure(detal_idx=delta_idx)
        t_mean=self.cal_mean(self.true)
        p_mean=self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean,pred_mean=p_mean)


    def train_test_model(self):
        self.xgboostmodel()
        self.test_model()

xgbm = XgboostModel(jobname='xgb_model_convt_temp_2',fc='wfzc',fj='A2',model_kind='convt_temp_2',max_depth=56,n_estimator=300)

xgbm.xgboostmodel()
xgbm.test_model()
