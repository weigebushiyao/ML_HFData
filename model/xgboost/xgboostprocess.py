# -*-coding:utf-8-*-
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import time
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from model.get_data_path import get_filtered_data_path, get_test_data_path, get_train_data_path
from sklearn.model_selection import train_test_split
import os
from util.show_save_result import ShowAndSave
import logging


cur_path = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
riqi = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
log_path = cur_path + '/logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_name = log_path + riqi + '.log'
fh = logging.FileHandler(log_name, mode='w')
formatter = logging.Formatter("%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s:%(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


class XgboostModel(ShowAndSave):
    def __init__(self, params=None, jobname='xgb_model', fc='wfzc', fj='A2', model_kind='cap_temp_1', max_depth=20,
                 n_estimator=512, min_child_weight=3, params_kind=None):
        super().__init__()
        self.job_name = jobname + '_' + fc + '_' + fj + '_' + str(max_depth) + '_' + str(n_estimator) + '_' + str(
            min_child_weight)
        logger.info(self.job_name)
        self.model_folder_name = fj + '_' + model_kind + '_' + params_kind
        self.model_name = fj
        self.model_kind = model_kind
        self.params_kind = params_kind
        self.fc = fc
        self.fj = fj
        self.model_kind = model_kind
        self.cur_path = cur_path
        self.init_param()
        self.params = params
        self.fj_model_kind = fj + '_' + model_kind
        self.model_file_name = self.fj_model_kind + '_xgb.model'
        self.feature_importance_path = self.single_model_path + 'feature_importance/'
        self.max_depth = max_depth
        self.n_estimator = n_estimator
        self.min_child_weight = min_child_weight

    def xgboostmodel(self, data_kind='train_data'):
        self.data_kind = data_kind
        data_file = get_train_data_path(self.fc, self.fj, self.model_kind, self.params_kind)
        df = pd.read_csv(data_file, encoding='utf-8', index_col=0, low_memory=False)
        logger.info(df.columns)
        logger.info(df.shape)
        traindata = df.iloc[:, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)  # list
        logger.info('when training,train data number:{}'.format(str(len(y_train))))
        logger.info('when training,test data number:{}'.format(len(y_test)))
        logger.info('training model:{}'.format(self.model_kind))
        # params={'booster':'gbtree','objective':'reg:squarederror','eval_metric':'rmse','seed':0,'n_jobs':10,'max_depth':self.max_depth,'n_estimators':self.n_estimator,'min_child_weight':self.min_child_weight,
        #         'verbosity':1,'learning_rate':0.05}
        raw_model = xgb.XGBRegressor(max_depth=self.max_depth,
                                     n_estimators=self.n_estimator, learning_rate=0.02, silent=False,
                                     min_child_weight=self.min_child_weight,tree_mothod='gpu_hist')
        # raw_model = xgb.XGBRegressor(**params)
        raw_model.fit(x_train, y_train)
        logger.info(self.model_path)
        raw_model.save_model(self.model_path + self.model_file_name)
        pred = raw_model.predict(x_test)
        plot_importance(raw_model)
        if not os.path.exists(self.feature_importance_path):
            os.makedirs(self.feature_importance_path)
        plt.savefig(self.feature_importance_path + self.fj_model_kind + '_feature_importance')
        plt.show()
        plt.close()
        self.save_result_dataframe(y_test, pred, )
        self.set_var(true_v=y_test, pred_v=pred)
        self.show_save_figure(detal_idx=4)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean, train_n=len(x_train), test_n=len(x_test))

    def test_model(self, data_kind='fault_data', delta_idx=2):
        self.data_kind = data_kind
        fault_test_file_path = get_test_data_path(self.fc, self.fj, self.model_kind, self.params_kind)
        df = pd.read_csv(fault_test_file_path, encoding='utf-8', index_col=0)
        data = df.iloc[:, :].values
        x = data[:, :-1]
        y = data[:, -1]
        xgbr = xgb.XGBRegressor()
        logger.info(self.model_path + self.model_file_name)
        xgbr.load_model(self.model_path + self.model_file_name)
        pred = xgbr.predict(x)
        self.save_result_dataframe(y, pred)
        self.set_var(true_v=y, pred_v=pred)
        # self.show_rolling_fig()
        self.show_save_figure(detal_idx=delta_idx)
        t_mean = self.cal_mean(self.true)
        p_mean = self.cal_mean(self.pred)
        self.save_result(true_mean=t_mean, pred_mean=p_mean)

    def params_tuned(self):
        xgbr = xgb.XGBRegressor(objective='reg:squarederror')
        datafile = get_filtered_data_path(self.fc, self.fj, self.model_kind, self.params_kind)
        params = {'max_depth': [16, 32, 48], 'n_estimators': [128, 256, 512], 'min_child_weight': [3]}
        grid = RandomizedSearchCV(xgbr, params, cv=3, scoring='neg_mean_squared_error', n_iter=6)
        df = pd.read_csv(datafile, encoding='utf-8', index_col=0)
        traindata = df.iloc[100000:250000, :].values
        x = traindata[:, :-1]
        y = traindata[:, -1]
        grid.fit(x, y)
        logger.info(grid.best_score_)
        logger.info(grid.best_params_)
        self.params = grid.best_params_
        df = pd.DataFrame(list(self.params.items()))
        df.to_csv(self.params_file_path + 'params.csv', encoding='utf-8')

    def train_test_model(self):
        self.xgboostmodel()
        self.test_model()

    def predict(self,x):
        xgbr = xgb.XGBRegressor()
        #logger.info(self.model_path + self.model_file_name)
        xgbr.load_model(self.model_path + self.model_file_name)
        pred = xgbr.predict(x)
        return pred


xgbm = XgboostModel(jobname='xgb_model_motor', fc='wfzc', fj='A2', model_kind='motor_temp_1',
                    params_kind='model_params_v3', max_depth=32, n_estimator=350, min_child_weight=1)
# xgbm.params_tuned()
xgbm.xgboostmodel()
xgbm.test_model()
