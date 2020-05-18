import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from model.get_data_path import get_train_data_path,get_test_data_path
from sklearn.model_selection import train_test_split
import os
from util.show_save_result import ShowAndSave
from sklearn.externals import joblib

cur_path = os.path.abspath(os.path.dirname(__file__))



class AdaboostModel(ShowAndSave):
    def __init__(self, params=None, jobname='adb_model', fc='wfzc', fj='A2', model_kind='cap_temp_1', max_depth=8,
                 n_estimator=512):
        super().__init__()
        self.job_name = jobname + '_' + fc + '_' + fj + '_' + str(max_depth) + '_' + str(n_estimator)
        self.model_folder_name = fj + '_' + model_kind
        self.model_name = fj
        self.model_kind = model_kind
        self.fc = fc
        self.fj = fj
        self.model_kind = model_kind
        self.cur_path = cur_path
        self.init_param()
        self.params = params
        self.fj_model_kind = fj + '_' + model_kind
        self.model_file_name = self.fj_model_kind + '_adb.model'
        self.feature_importance_path = self.single_model_path + 'feature_importance/'
        self.max_depth = max_depth
        self.n_estimator = n_estimator


    def adaboostmodel(self,data_kind='train'):
        self.data_kind = data_kind
        data_file = get_train_data_path(self.fc, self.fj, self.model_kind)
        df = pd.read_csv(data_file, encoding='utf-8', index_col=0, low_memory=False)
        print(df.shape)
        traindata = df.iloc[:, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)  # list6
        print('when training,train data number:', len(y_train))
        print('when training,test data number:', len(y_test))
        print('training model:', self.model_kind)
        raw_model = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_features=None, max_depth=self.max_depth, min_samples_split=20,
                                                 min_samples_leaf=10, min_weight_fraction_leaf=0, max_leaf_nodes=None),
            learning_rate=0.01, loss='square', n_estimators=self.n_estimator)
        raw_model.fit(x_train, y_train)
        print(self.model_path)

        joblib.dump(raw_model,self.model_path + self.model_file_name)
        pred = raw_model.predict(x_test)

        self.save_result_dataframe(y_test, pred, )
        self.set_var(true_v=y_test, pred_v=pred)
        self.show_save_figure(detal_idx=4)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean, train_n=len(x_train), test_n=len(x_test))


    def test_model(self):
        fault_test_file_path=get_test_data_path(self.fc, self.fj, self.model_kind)
        df=pd.read_csv(fault_test_file_path,encoding='utf-8',index_col=0)
        data=df.iloc[:,:].values
        x=data[:,:-1]
        y=data[:,-1]
        raw_model=joblib.load(self.model_path+self.model_file_name)
        pred=raw_model.predict(x)
        self.true=y
        self.pred=pred
        self.show_save_figure(detal_idx=4)
        t_mean=self.cal_mean(self.true)
        p_mean=self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean,pred_mean=p_mean)


adb = AdaboostModel(jobname='adb_model', fc='wfzc', fj='A2', model_kind='convt_temp_2', max_depth=15,
                 n_estimator=512)
adb.adaboostmodel()
adb.test_model()