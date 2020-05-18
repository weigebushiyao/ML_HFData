import pandas as pd
import os
from model.get_data_path import get_train_data_path,get_filtered_data_path
from model.params_dict import ParamsDict

class FilterAbnormalValue():

    def __init__(self, fc,fj,model_kind,k=2.5,params_kind=None):
        self.fc = fc  # 风电场
        self._time_list=None#故障时间
        self.fj_YearMonth=set()
        self.fj=fj
        self.model_kind=model_kind
        self.params_kind=params_kind
        self.k=k
        cur_path=os.path.abspath(os.path.dirname(__file__))
        self.data_path= os.path.abspath(os.path.join(cur_path, '..')) + '/data/'
        self.mergedData_filtered_path=self.data_path+'mergedData_filtered/'+fc+'_'+fj+'_'+model_kind+'_'+params_kind+'/'
        self.filtered_data_file=fj+'_'+model_kind+'_filtered.csv'
        if not os.path.exists(self.mergedData_filtered_path):
            os.makedirs(self.mergedData_filtered_path)
        model_params = ParamsDict()
        print(model_kind)
        self.params = model_params.model_params[params_kind][model_kind]

    def filter_abnormal_data(self,feature_name):
        data_file=get_filtered_data_path(self.fc,self.fj,self.model_kind,self.params_kind)
        df = pd.read_csv(data_file, encoding='utf-8', index_col=0)
        des = df.describe()
        print(des)
        print(df.shape)
        print(feature_name)
        label_mean = des.loc['mean', feature_name]
        label_std = des.loc['std', feature_name]
        max_abnormal_val = label_mean + self.k * label_std
        min_abnormal_val = label_mean - self.k * label_std
        print(max_abnormal_val, min_abnormal_val)
        filtered_df_min=df[min_abnormal_val<df[feature_name]]
        filtered_df = filtered_df_min[filtered_df_min[feature_name] < max_abnormal_val]
        print(filtered_df.shape)
        if os.path.exists(self.mergedData_filtered_path+self.filtered_data_file):
            print('remove file:'+self.mergedData_filtered_path+self.filtered_data_file)
            os.remove(self.mergedData_filtered_path+self.filtered_data_file)
        filtered_df.to_csv(self.mergedData_filtered_path+self.filtered_data_file,encoding='utf-8')

    def feature_data_filter(self):
        data_file = get_train_data_path(self.fc, self.fj, self.model_kind,params_kind=self.params_kind)
        df = pd.read_csv(data_file, encoding='utf-8', index_col=0)
        print(df.columns)
        for f in self.params:
            df=self.recursive_filter_abnormal_data(df,f)
        if os.path.exists(self.mergedData_filtered_path+self.filtered_data_file):
            print('remove file:'+self.mergedData_filtered_path+self.filtered_data_file)
            os.remove(self.mergedData_filtered_path+self.filtered_data_file)
        df.to_csv(self.mergedData_filtered_path+self.filtered_data_file,encoding='utf-8')

    def recursive_filter_abnormal_data(self,df,feature_name):
        des = df.describe()
        print(des)
        print(df.shape)
        print(feature_name)
        label_mean = des.loc['mean', feature_name]
        label_std = des.loc['std', feature_name]
        max_abnormal_val = label_mean + self.k * label_std
        min_abnormal_val = label_mean - self.k * label_std
        print(max_abnormal_val, min_abnormal_val)
        filtered_df_min=df[min_abnormal_val<df[feature_name]]
        filtered_df = filtered_df_min[filtered_df_min[feature_name] < max_abnormal_val]
        print(filtered_df.shape)
        return filtered_df

fav=FilterAbnormalValue(fc='wfzc',fj='A2',model_kind='motor_temp_2',k=3,params_kind='model_params_v2')
fav.feature_data_filter()