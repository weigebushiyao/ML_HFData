#-*-coding=utf-8-*-
import pandas as pd
import os
import seaborn as sbn
import numpy as np
from util.featureselector import FeatureSelector
import time
from model.get_data_path import get_train_data_path,get_filtered_data_path
from model.params_dict import ParamsDict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
Feature importances will change on multiple runs of the machine learning model
Decide whether or not to keep the extra features created from one-hot encoding
Try out several different values for the various parameters to decide which ones work best for a machine learning task
The output of missing, single unique, and collinear will stay the same for the identical parameters
Feature selection is a critical step of a machine learning workflow that may require several iterations to optimize
"""
now_time=time.strftime('%Y%m%d')
hour_minute=time.strftime("%m%d%H%M")
cur_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(cur_path, '..'))
# data_path = parent_path + '/data/mergedData_new/'
# if not os.path.exists(data_path):
#     sys.exit(1)

class DataMining():
    def __init__(self,fc=None,fj=None,model_kind=None,params_kind=None):
        #self.dataFile = data_path + os.listdir(data_path)[0]
        self.dataFile=get_train_data_path(fc,fj,model_kind,params_kind)
        self.df = pd.read_csv(self.dataFile, encoding='utf-8', index_col=0)  # 省去索引
        result_path = cur_path + '/result/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.single_result_path = result_path + 'result_' +fc+'_'+fj+'_'+model_kind+ '/'
        if not os.path.exists(self.single_result_path):
            os.makedirs(self.single_result_path)
        self.figure_path=self.single_result_path+'_'+params_kind+'_'+ hour_minute + '/'

        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)
        self.model_number=str.split(model_kind,'_')[-1]
        self.params=ParamsDict().model_params[params_kind][model_kind]


    def feature_describe(self):
        des = self.df.describe()
        print(des)
        min_val=des.loc['min']
        max_val=des.loc['max']
        std_val=des.loc['std']
        mean_val=des.loc['mean']
        return min_val,max_val,std_val,mean_val

    def dataAnalysis(self):

        corr=self.df.corr()
        corr.to_csv(self.single_result_path+'pearson_correlation_result.csv',encoding='utf-8')
        des=self.df.describe()
        print(des)
        label_mean=des.loc['mean',self.params[-1]]
        label_std=des.loc['std',self.params[-1]]
        max_abnormal_val=label_mean+2.5*label_std
        min_abnormal_val=label_mean-2.5*label_std
        print(max_abnormal_val,min_abnormal_val)
        des.to_csv(self.single_result_path+'data_analysis_result.csv',encoding='utf-8')
        #print(des)
        print(self.df.columns.values)
        self.df.dropna()
        #df = df[9990:200000]
        print(self.df.shape)
        train_label = self.df[self.params[-1]]
        print(train_label)
        #df = self.df.drop(columns=[self.params[:-1]])
        df=self.df.iloc[:,:-1]
        fs = FeatureSelector(data=df, labels=train_label)
        fs.figure_path=self.figure_path
        self.missingValueAnalysis(fs)
        self.singleValueAnalysis(fs)
        self.collinearFeatureAnalysis(fs,thr=0.9)
        fs.identify_zero_importance(task='regression',eval_metric='L2',n_iterations=6,early_stopping=True)
        one_hot_features=fs.one_hot_features
        base_features=fs.base_features
        print("There are %d original features" % len(base_features))
        print('There are %d one-hot feature' % len(one_hot_features))
        fs.data_all.head()
        zeroimportancefeature=fs.ops['zero_importance']
        print(zeroimportancefeature)

        fs.plot_feature_importances(threshold=0.9,plot_n=10)
        fs.feature_importances.head(9)

        fs.identify_low_importance(cumulative_importance=0.9)
        lowimportancefeatures=fs.ops['low_importance']
        lowimportancefeatures[:5]

        removemissingvalue=fs.remove(methods=['missing'])
        removezeroimportance=fs.remove(methods=['missing','zero_importance'])
        alltoremoved=fs.check_removal()
        print(alltoremoved)
        #简便的进行特征筛选
        dataremoved=fs.remove(methods='all')
        dataremovedall=fs.remove(methods='all',keep_one_hot=False)
        # fs = FeatureSelector(data=df, labels=train_label)
        #
        # fs.identify_all(selection_params={'missing_threshold': 0.6, 'correlation_threshold': 0.98,
        #                                   'task': 'classification', 'eval_metric': 'L2',
        #                                   'cumulative_importance': 0.99})



   # def lowImportanceFeatures(self):

    def zeroImportanceFeature(self,fs,):
        fs.identify_zero_importance(task='regresssion', eval_metric='auc', n_iterations=20, early_stopping=True)
        one_hot_features = fs.one_hot_features
        base_features = fs.base_features
        print("There are %d original features" % len(base_features))
        print('There are %d one-hot feature' % len(one_hot_features))
        fs.data_all.head()
        zeroimportancefeature = fs.ops['zero_importance']
        print(zeroimportancefeature)

    def collinearFeatureAnalysis(self,fs,thr):
        fs.identify_collinear(correlation_threshold=thr)
        correlated_features = fs.ops['collinear']
        print(correlated_features)
        fs.plot_collinear()
        fs.plot_collinear(plot_all=True)
        fs.record_collinear.head()

    def singleValueAnalysis(self,fs):
        fs.identify_single_unique()
        single_unique = fs.ops['single_unique']
        print(single_unique)
        fs.plot_unique()
        fs.unique_stats.sample(5)

    def missingValueAnalysis(self,fs):
        fs.identify_missing(missing_threshold=0.6)
        missing_features = fs.ops['missing']
        print(missing_features)
        fs.plot_missing()
        fs.missing_stats.head(10)

    def data_decomposition_pac(self):
        traindata = self.df.iloc[:, :].values
        x = traindata[:, :-1]
        pca=PCA(n_components=1)
        new_x=pca.fit_transform(x)
        #plt.scatter(new_x,y,marker='o',color='green',s=20)
        #plt.plot(new_x,y,color="green")
        plt.hist(new_x,bins=1500)
        #plt.hist(y)
        plt.show()
        plt.close()

    def plot_features(self):
        min_val,max_val,_,_=self.feature_describe()
        min_val_feat_list=min_val.values
        feature_name=min_val.index
        max_val_feat_list=max_val.values
        traindata = self.df.iloc[:, :].values
        num_features=len(feature_name)
        import math
        for i in range(num_features):
            feat_name=feature_name[i]
            min_=min_val_feat_list[i]
            max_=max_val_feat_list[i]
            val_bins=math.floor(max_-min_)*10
            x = traindata[:, i]
            plt.xlabel(feat_name)
            plt.hist(x,bins=val_bins)
            plt.savefig(self.figure_path+feat_name+'_hist.png')
            plt.show()
            #plt.close()

    def plot_data_line(self):
        min_val, max_val, _, _ = self.feature_describe()
        feature_name = min_val.index
        traindata = self.df.iloc[:, :].values
        num_features = len(feature_name)
        for i in range(num_features):
            feat_name = feature_name[i]
            y = traindata[:, i]
            x=np.arange(0,len(y),1)
            plt.xlabel(feat_name)
            plt.plot(x,y)
            plt.savefig(self.figure_path + feat_name + '_line.png')
            plt.show()
            # plt.close()


dm = DataMining(fc='wfzc',fj='A2',model_kind='motor_temp_1',params_kind='model_params_v2')
# dm.dataAnalysis()
# #dm.data_decomposition_pac()
# #dm.feature_describe()
dm.plot_features()
#dm.plot_data_line()