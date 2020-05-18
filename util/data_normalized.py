import numpy as np
# from model.get_data_path import get_train_data_path
# import pandas as pd

def data_normalized(dataset):
    n_features=np.shape(dataset)[1]
    print(n_features)
    tmp_list=[]
    mean_std_dict=[]
    for i in range(n_features):
        data=dataset[:,i]
        mean=np.mean(data,axis=0)
        std=np.std(data,axis=0)
        normalized_data=(data-mean)/std
        tmp_list.append(np.array(normalized_data).reshape(-1,1))
        mean_std_dict.append({'mean':mean,'std':std})
    res=np.concatenate((np.array(tmp_list)),axis=1)
    print(mean_std_dict)
    return res
# filename=get_train_data_path()
# df=pd.read_csv(filename,encoding='utf-8',index_col=0)
# dataset=df.iloc[:,:].values
# data=dataset[:,:-1]
# data=np.array(data)
# _=data_normalized(data)

# dataset=[[1,2,3],[2,3,4]]
# dataset=np.array(dataset)
# mean=np.mean(dataset)
# std=np.std(dataset)
# print(mean)
# tmp=(dataset-mean)/std
# print(tmp)
# print('________')
# std=np.std(dataset)
# res=data_normalized(dataset)
# print(res)